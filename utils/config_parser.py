import configparser

class ConfigParser:
    def __init__(self, config_path):
        self.parser = configparser.ConfigParser()
        self.parser.read(config_path)

    def get_section(self, section_name): # Get all the key-value pairs in a section
        if section_name not in self.parser: # Check if section exists
            raise ValueError(f"Section {section_name} not found in config file") # Raise error if section does not exist
        return dict(self.parser[section_name])

    def get_all_sections(self): # Get all the sections in the config file
        return {
            section:
                dict(self.parser[section]) for section in self.parser.sections()
        }