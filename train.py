from MS2.train_apply import train_waveunet, create_arg_parser

if __name__ == '__main__':
    parser = create_arg_parser()
    train_waveunet(parser.parse_args())
