import argparse


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()

    CLI.add_argument(
        "--config_id", type=str, help='Configuration ID for training'

    )

    ARGS = CLI.parse_args()

    config_id = ARGS.config_id

    print('Make Dataset !!!!')