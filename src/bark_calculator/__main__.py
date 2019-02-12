import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--good_examples", help="Run with pre-filtered good examples",
                        action="store_true")

    args = parser.parse_args()

    print("Good_examples mode {}".format("on" if args.good_examples else "off"))

