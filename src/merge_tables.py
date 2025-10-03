from config.config import CONFIG
from library.TableCreator.generate_tab_dataset import GenerateDataset



if __name__ == '__main__':
    create_tab = GenerateDataset(path_tables=CONFIG.get("paths")['path_tables'],
                                 output_path= CONFIG.get("paths")['outputs_tabs'])
    create_tab.run()















