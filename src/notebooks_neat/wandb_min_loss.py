# This script will update summary statistic of wandb runs

import wandb

# run_path = "vh-motion-pred/rnn/3u9td4jj"
api = wandb.Api()
# run = api.run(run_path)

runs = api.runs('rnn')

metric = 'epoch_val_loss'
for i, run in enumerate(runs):
    print(run.name, run.id)
    df = run.history(samples=40000)
    if metric in df:
        # for some reason where it's not recorded we get nan (float?), and where it was invalid we get 'NaN'.
        # If this is empty we get nan out which is ok.
        min_metric = df.loc[df[metric].notnull() & (df[metric] != 'NaN'), 'epoch_val_loss'].min()
        run.summary['min_' + metric] = min_metric
        run.summary.update()
