from argparse import Namespace
from typing import List
import torch
import torch.nn as nn
import numpy as np
from featurization import get_atom_fdim, get_bond_fdim, mol2graph
import math
import torch.nn.functional as F
from torch_scatter import scatter_add



def attention(query, key, value, mask, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class AttentionLayer(nn.Module):
    def __init__(self, args):
        super(AttentionLayer, self).__init__()
        self.hidden_size = args.hidden_size
        self.w_q = nn.Linear(133, 32)
        self.w_k = nn.Linear(133, 32)
        self.w_v = nn.Linear(133, 32)

        self.dense = nn.Linear(32, 133)
        self.LayerNorm = nn.LayerNorm(133, eps=1e-6)
        self.dropout = nn.Dropout(0.1)

    def forward(self, fg_hiddens, init_hiddens):
        query = self.w_q(fg_hiddens)
        key = self.w_k(fg_hiddens)
        value = self.w_v(fg_hiddens)

        padding_mask = (init_hiddens != 0) + 0.0
        mask = torch.matmul(padding_mask, padding_mask.transpose(-2, -1))
        x, attn = attention(query, key, value, mask)

        hidden_states = self.dense(x)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + fg_hiddens)

        return hidden_states


class Prompt_generator(nn.Module):
    def __init__(self, args):
        super(Prompt_generator, self).__init__()
        self.hidden_size = args.hidden_size
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.1)
        self.cls = nn.Parameter(torch.randn(1, 133), requires_grad=True)
        self.linear = nn.Linear(133, self.hidden_size)
        self.attention_layer_1 = AttentionLayer(args)
        self.attention_layer_2 = AttentionLayer(args)
        self.norm = nn.LayerNorm(args.hidden_size)

    def forward(self, atom_hiddens: torch.Tensor, fg_states: torch.Tensor, atom_num, fg_indexs):
        for i in range(len(fg_indexs)):
            fg_states.scatter_(0, fg_indexs[i:i + 1], self.cls)

        hidden_states = self.attention_layer_1(fg_states, fg_states)
        hidden_states = self.attention_layer_2(hidden_states, fg_states)
        fg_out = torch.zeros(1, self.hidden_size).cuda()
        cls_hiddens = torch.gather(hidden_states, 0, fg_indexs)
        cls_hiddens = self.linear(cls_hiddens)
        fg_hiddens = torch.repeat_interleave(cls_hiddens, torch.tensor(atom_num).cuda(), dim=0)
        fg_out = torch.cat((fg_out, fg_hiddens), 0)

        fg_out = self.norm(fg_out)
        return atom_hiddens + self.alpha * fg_out


class PromptGeneratorOutput(nn.Module):
    def __init__(self, args, self_output):
        super(PromptGeneratorOutput, self).__init__()
        # change position
        self.self_out = self_output
        self.prompt_generator = Prompt_generator(args)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.self_out(hidden_states)
        return hidden_states


def prompt_generator_output(args):
    return lambda self_output: PromptGeneratorOutput(args, self_output)


def add_functional_prompt(model, args):
    model.encoder.encoder.W_i_atom = prompt_generator_output(args)(model.encoder.encoder.W_i_atom)
    return model


def build_pretrain_model(args: Namespace, encoder_name) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    args.ffn_hidden_size = args.hidden_size // 2
    args.output_size = args.hidden_size

    model = MoleculeModel(classification=args.dataset_type == 'classification',
                          multiclass=args.dataset_type == 'multiclass', pretrain=True)
    model.create_encoder(args, encoder_name)
    model.create_ffn(args)

    initialize_weights(model)

    return model


def build_model(args: Namespace, encoder_name) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass', pretrain=False)
    model.create_encoder(args, encoder_name)
    model.create_ffn(args)

    initialize_weights(model)

    return model


def initialize_weights(model: nn.Module):
    """
    Initializes the weights of a model in place.

    :param model: An nn.Module.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)


def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == 'GELU':
        return nn.GELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.

    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target[index == 0] = 0
    return target


class CMPNEncoder(nn.Module):
    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        super(CMPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Input
        input_dim = self.atom_fdim
        self.W_i_atom = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        input_dim = self.bond_fdim
        self.W_i_bond = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        w_h_input_size_atom = self.hidden_size + self.bond_fdim
        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)

        w_h_input_size_bond = self.hidden_size

        for depth in range(self.depth - 1):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear((self.hidden_size) * 2, self.hidden_size)

        self.gru = BatchGRU(self.hidden_size)

        self.lr = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=self.bias)

        # add & concat functional group features
        if self.args.step == 'functional_prompt':
            self.cls = nn.Parameter(torch.randn(1, 133), requires_grad=True)
            self.W_i_atom_new = nn.Linear(self.atom_fdim * 2, self.hidden_size, bias=self.bias)
        elif self.args.step == 'functional_concat':
            self.cls = nn.Parameter(torch.randn(1, 133), requires_grad=True)
            self.W_i_atom_new = nn.Linear(self.atom_fdim+133, self.hidden_size, bias=self.bias)

    def forward(self, step, mol_graph, features_batch=None) -> torch.FloatTensor:

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, atom_num, fg_num, f_fgs, fg_scope = mol_graph.get_components()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb, f_fgs = (
                f_atoms.cuda(), f_bonds.cuda(),
                a2b.cuda(), b2a.cuda(), b2revb.cuda(), f_fgs.cuda())

        fg_index = [i * 13 for i in range(mol_graph.n_mols)]
        fg_indxs = [[i] * 133 for i in fg_index]
        fg_indxs = torch.LongTensor(fg_indxs).cuda()

        if self.args.step == 'functional_prompt':
            # make sure the prompt exists
            assert self.W_i_atom.prompt_generator
            # Input
            input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
            input_atom = self.W_i_atom.prompt_generator(input_atom, f_fgs, atom_num, fg_indxs)

        elif self.args.step == 'functional_concat':
            for i in range(len(fg_indxs)):
                f_fgs.scatter_(0, fg_indxs[i:i + 1], self.cls)

            target_index = [val for val in range(mol_graph.n_mols) for i in range(13)]
            target_index = torch.LongTensor(target_index).cuda()
            fg_hiddens = scatter_add(f_fgs, target_index, 0)
            fg_hiddens_atom = torch.repeat_interleave(fg_hiddens, torch.tensor(atom_num).cuda(), dim=0)
            fg_out = torch.zeros(1, 133).cuda()
            fg_out = torch.cat((fg_out, fg_hiddens_atom), 0)
            f_atoms = torch.cat((fg_out, f_atoms), 1)
            # Input
            input_atom = self.W_i_atom_new(f_atoms)  # num_atoms x hidden_size
        elif self.args.step == 'none':
            # Input
            input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
        else:
            raise ValueError

        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()

        input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)

        # Message passing
        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)
            # agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            agg_message = agg_message.mean(dim=1)
            message_atom = message_atom + agg_message

            # directed graph
            rev_message = message_bond[b2revb]  # num_bonds x hidden
            message_bond = message_atom[b2a] - rev_message  # num_bonds x hidden

            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))

        agg_message = index_select_ND(message_bond, a2b)
        # agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
        agg_message = agg_message.mean(dim=1)
        agg_message = self.lr(torch.cat([agg_message, message_atom, input_atom], 1))
        agg_message = self.gru(agg_message, a_scope)

        atom_hiddens = self.act_func(self.W_o(agg_message))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))

        mol_vecs = torch.stack(mol_vecs, dim=0)
        # print(mol_vecs)
        return mol_vecs  # B x H


class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True,
                          bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size),
                                1.0 / math.sqrt(self.hidden_size))

    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))

            cur_message = torch.nn.ZeroPad2d((0, 0, 0, MAX_atom_len - cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))

        message_lst = torch.cat(message_lst, 0)
        hidden_lst = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2, 1, 1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)

        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2 * self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)

        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1),
                             cur_message_unpadding], 0)
        return message


class CMPN(nn.Module):
    def __init__(self,
                 args: Namespace,
                 func_smi_to_repr=None,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        super(CMPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + \
                         (not args.atom_messages) * self.atom_fdim  # * 2
        self.graph_input = graph_input
        self.encoder = CMPNEncoder(self.args, self.atom_fdim, self.bond_fdim)
        self.func_smi_to_repr = func_smi_to_repr

    def forward(self, step, prompt: bool, batch, features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args, prompt, self.func_smi_to_repr)
        # f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, atom_num, fg_num, f_fgs, fg_scope = batch.get_components()
        output = self.encoder.forward(step, batch, features_batch)
        return output


class MoleculeModel(nn.Module):
    def __init__(self, classification: bool, multiclass: bool, pretrain: bool):
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)
        self.pretrain = pretrain

    def create_encoder(self, args, encoder_name, func=None):
        if encoder_name == 'CMPNN':
            self.encoder = CMPN(args, func)

    def create_ffn(self, args):
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes

        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size * 1
            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)


    def forward(self, *input):
        if not self.pretrain:
            output = self.ffn(self.encoder(*input))

            # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
            if self.classification and not self.training:
                output = self.sigmoid(output)
            if self.multiclass:
                output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
                if not self.training:
                    output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
        else:
            output = self.ffn(self.encoder(*input))

        return output