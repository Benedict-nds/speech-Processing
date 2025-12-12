from code.data.loader import DAICWOZLoader

def main():
    loader = DAICWOZLoader()
    metadata = loader.build_metadata()
    print(metadata.head())

if __name__ == "__main__":
    main()