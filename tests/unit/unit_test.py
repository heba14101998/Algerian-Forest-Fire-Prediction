from src.utils import read_yaml

CONFIGS, _ = read_yaml("params.yaml")

class TestDataPreprocessor(unittest.TestCase):
    def SetUp(self):
        
        self.configs = DataPreprocessor(CONFIGS)
        # Load the actual dataset for testing
        self.configs.data = pd.read_csv(
            os.path.join(CONFIGS.raw_data_dir, CONFIGS.data_file_name))

    def test_missing_values():
        pass
        # assertEqual(cleaned_data.isnull().sum().sum(), 0)

    def test_target_column_values():
        pass


    def test_saving_files():
        pass


    def test_num_of_features():
        pass
