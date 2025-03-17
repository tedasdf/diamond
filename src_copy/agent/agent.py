import wandb


if __name__ == "__main__":
    wandb.login(key="272a6329f05b645580139131ec5b1eb14bb27769")
    wandb.init(

        project="RL_pacman",
        
        )