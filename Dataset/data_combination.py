# environment: PhD-RC; combine the data from different sources
import cirpy
import pandas as pd
import numpy as np

'''OH'''
# name_file = 'OH'
# smiles_all = []
# LogK_all = []
# source = []
# process_info = []
# for sheet in ['Sheet1', 'Sheet2', 'Sheet3', 'Sheet4', 'Sheet5', 'Sheet6']:  # Sheet1-5
#     data = pd.read_excel(f'dataset-{name_file}.xlsx', sheet_name=sheet)
#     if sheet in ['Sheet1', 'Sheet2', 'Sheet3', 'Sheet6']:
#         smiles_all.extend(list(data['Smiles']))
#         LogK_all.extend(list(data['LogK']))
#         source.extend([sheet for _ in range(len(list(data['LogK'])))])
#     elif sheet in ['Sheet4', 'Sheet5']:
#         name_list = list(data['Name'])
#         logK = list(data['LogK'])
#         for idx, name in enumerate(name_list):
#             try:
#                 smi = cirpy.resolve(name, 'smiles')
#                 print(f'Name: {name} was converted to Smiles {smi}')
#                 process_info.append(f'Name: {name} was converted to Smiles {smi}')
#                 smiles_all.append(smi)
#                 LogK_all.append(logK[idx])
#                 source.append(sheet)
#             except:
#                 pass
#     else:
#         raise ValueError(f'Do not have this sheet {sheet}')
# np.savetxt('OH-combination.txt', np.array([smiles_all, LogK_all, source]).T, fmt='%s')
# np.savetxt('OH-process_info.txt', process_info, fmt='%s')

'''SO4-'''
# name_file = 'SO4-'
# smiles_all = []
# LogK_all = []
# source = []
# process_info = []
# for sheet in ['Sheet1', 'Sheet2', 'Sheet3']:  #
#     data = pd.read_excel(f'dataset-{name_file}.xlsx', sheet_name=sheet)
#     if sheet in ['Sheet1', 'Sheet3']:
#         smiles_all.extend(list(data['Smiles']))
#         LogK_all.extend(list(data['LogK']))
#         source.extend([sheet for _ in range(len(list(data['LogK'])))])
#     elif sheet in ['Sheet2']:
#         name_list = list(data['Name'])
#         logK = list(data['LogK'])
#         for idx, name in enumerate(name_list):
#             smi = cirpy.resolve(name, 'smiles')
#             print(f'Name: {name} was converted to Smiles {smi}')
#             process_info.append(f'Name: {name} was converted to Smiles {smi}')
#             smiles_all.append(smi)
#             LogK_all.append(logK[idx])
#             source.append(sheet)
#     else:
#         raise ValueError(f'Do not have this sheet {sheet}')
# np.savetxt('SO4--combination.txt', np.array([smiles_all, LogK_all, source]).T, fmt='%s')
# np.savetxt('SO4--process_info.txt', process_info, fmt='%s')

'''O3'''
# name_file = 'O3'
# smiles_all = []
# LogK_all = []
# source = []
# process_info = []
# for sheet in ['Sheet1', 'Sheet3']:
#     data = pd.read_excel(f'dataset-{name_file}.xlsx', sheet_name=sheet)
#     if sheet in ['Sheet2']:
#         smiles_all.extend(list(data['Smiles']))
#         LogK_all.extend(list(data['LogK']))
#         source.extend([sheet for _ in range(len(list(data['LogK'])))])
#     elif sheet in ['Sheet1']:
#         name_list = list(data['Name'])
#         logK = list(data['LogK'])
#         for idx, name in enumerate(name_list):
#             smi = cirpy.resolve(name, 'smiles')
#             print(f'Name: {name} was converted to Smiles {smi}')
#             process_info.append(f'Name: {name} was converted to Smiles {smi}')
#             smiles_all.append(smi)
#             LogK_all.append(logK[idx])
#             source.append(sheet)
#     elif sheet in ['Sheet3']:
#         name_list = list(data['Name'])
#         logK = list(data['LogK'])
#         for idx, name in enumerate(name_list):
#             name = name[1:-1]
#             smi = cirpy.resolve(name, 'smiles')
#             print(f'Name: {name} was converted to Smiles {smi}')
#             process_info.append(f'Name: {name} was converted to Smiles {smi}')
#             smiles_all.append(smi)
#             LogK_all.append(logK[idx])
#             source.append(sheet)
#     else:
#         raise ValueError(f'Do not have this sheet {sheet}')
# np.savetxt('O3-combination.txt', np.array([smiles_all, LogK_all, source]).T, fmt='%s')
# np.savetxt('O3-process_info.txt', process_info, fmt='%s')

'''Fe(VI)'''
# name_file = 'Fe(VI)'
# smiles_all = []
# LogK_all = []
# source = []
# process_info = []
# for sheet in ['Sheet1']:
#     data = pd.read_excel(f'dataset-{name_file}.xlsx', sheet_name=sheet)
#     if sheet in ['Sheet1']:
#         smiles_all.extend(list(data['Smiles']))
#         LogK_all.extend(list(data['LogK']))
#         source.extend([sheet for _ in range(len(list(data['LogK'])))])
#     else:
#         raise ValueError(f'Do not have this sheet {sheet}')
# np.savetxt('Fe(VI)-combination.txt', np.array([smiles_all, LogK_all, source]).T, fmt='%s')

'''HClO'''
# name_file = 'HClO'
# smiles_all = []
# LogK_all = []
# source = []
# process_info = []
# for sheet in ['Sheet1', 'Sheet2']:
#     data = pd.read_excel(f'dataset-{name_file}.xlsx', sheet_name=sheet)
#     if sheet in ['Sheet1', 'Sheet2']:
#         smiles_all.extend(list(data['Smiles']))
#         LogK_all.extend(list(data['LogK']))
#         source.extend([sheet for _ in range(len(list(data['LogK'])))])
#     else:
#         raise ValueError(f'Do not have this sheet {sheet}')
# np.savetxt('HClO-combination.txt', np.array([smiles_all, LogK_all, source]).T, fmt='%s')
# np.savetxt('HClO-process_info.txt', process_info, fmt='%s')


'''1O2'''
name_file = '1O2'
smiles_all = []
LogK_all = []
source = []
process_info = []
for sheet in ['Sheet1', 'Sheet2', 'Sheet3']:
    data = pd.read_excel(f'dataset-{name_file}.xlsx', sheet_name=sheet)
    if sheet in ['Sheet3']:
        smiles_all.extend(list(data['Smiles']))
        LogK_all.extend(list(data['LogK']))
        source.extend([sheet for _ in range(len(list(data['LogK'])))])
    elif sheet in ['Sheet1', 'Sheet2']:
        name_list = list(data['Name'])
        logK = list(data['LogK'])
        for idx, name in enumerate(name_list):
            smi = cirpy.resolve(name, 'smiles')
            print(f'Name: {name} was converted to Smiles {smi}')
            process_info.append(f'Name: {name} was converted to Smiles {smi}')
            smiles_all.append(smi)
            LogK_all.append(logK[idx])
            source.append(sheet)
    else:
        raise ValueError(f'Do not have this sheet {sheet}')
np.savetxt('1O2-combination.txt', np.array([smiles_all, LogK_all, source]).T, fmt='%s')