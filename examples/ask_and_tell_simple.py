import optuna


def f(x):
    return (x - 2) ** 2


def df(x):
    return 2 * x - 4


if __name__ == "__main__":
    study = optuna.create_study()

    for trial in study.ask():
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)

        x = 3
        for step in range(128):
            y = f(x)

            trial.report(y, step=step)
            if trial.should_prune():
                study.tell(trial, state="pruned")
                break

            gy = df(x)
            x -= gy * lr
        else:
            study.tell(trial, state="completed", value=y)

        if trial.number > 100:
            break

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
