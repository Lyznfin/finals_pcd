import argparse
from strawberry import classify_strawberry
from vanilla import classify_vanilla

def main():
    parser = argparse.ArgumentParser(description="Classify eye as either normal or cataract")
    parser.add_argument("input_path", type=str, help="Path to the input image")
    parser.add_argument("flavor", type=str, help="vanilla or strawberry (development)")
    args = parser.parse_args()
    match args.flavor:
        case 'vanilla':
            classify_vanilla(args.input_path)
        case 'strawberry':
            classify_strawberry(args.input_path)
        case _:
            return f"Wrong command"
    # print(args.input_path)

if __name__ == "__main__":
    main()
    # example usage
    # py main.py _test_file/normal_1.png strawberry
    # py main.py _test_file/cataract_1.png vanilla
    # py main.py _test_file/cataract_1.png strawberry