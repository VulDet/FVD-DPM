import argparse

class Parser:

    def __init__(self):

        self.parser = argparse.ArgumentParser(description='DPM-GSP')
        self.set_arguments()

    def set_arguments(self):

        self.parser.add_argument('--config', type=str, default="config.yaml", help="Path of config file")

        self.parser.add_argument('--comment', type=str, default="", 
                                    help="A single line comment for the experiment")
        self.parser.add_argument('--seed', type=int, default=42)

        self.parser.add_argument("--local_rank", type=int, default=-1)

        self.parser.add_argument("--world_size", type=int, default=4)

        self.parser.add_argument("--dataset", type=str, required=True, help="Dataset")

        self.parser.add_argument("--do_train", type=str, default='train', help="Whether to run training.")


        

    def parse(self):

        args, unparsed = self.parser.parse_known_args()
        
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        
        return args