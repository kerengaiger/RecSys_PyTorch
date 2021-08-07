class EarlyStop:
    def __init__(self, early_stop, early_stop_measure):
        self.endure = 0
        self.early_stop = early_stop
        self.early_stop_measure = early_stop_measure

        self.best_epoch = None
        self.best_score = None
        self.is_final_train = False

    def initialize(self):
        self.best_epoch = None
        self.best_score = None

    def step(self, score, preds_df, preds_out, epoch):
        # Always continue (shoudl_stop=False) if early_stop is not used

        if self.early_stop_measure == 'all':
            # Early stop if 'every' measure doesn't improve
            # Save individual best score & epoch
            if self.best_score is None:
                best_score = score
                self.best_epoch = {m: epoch for m in best_score}
                not_updated = False
            else:
                not_updated = True
                for metric in self.best_score:
                    if score[metric] > self.best_score[metric]:
                        self.best_score[metric] = score[metric]
                        self.best_epoch[metric] = epoch
                        not_updated = False
        else:
            # Early stop if specific measure doesn't improve
            # Save best score & epoch at the best epoch of the standard measure
            if self.best_score is None:
                self.best_score = score
                self.best_epoch = epoch
                if self.is_final_train:
                    preds_df.to_csv(preds_out, index=False)
                not_updated = False
            else:
                if score[self.early_stop_measure] > self.best_score[self.early_stop_measure]:
                    self.best_epoch = epoch
                    self.best_score = score
                    if self.is_final_train:
                        preds_df.to_csv(preds_out, index=False)
                    not_updated = False
                else:
                    not_updated = True

        should_stop = False
        if not_updated:
            self.endure += 1
            if self.early_stop and self.endure >= self.early_stop:
                should_stop = True
        else:
            self.endure = 0
            should_stop = False
        
        if self.early_stop < 1:
            should_stop = False

        return not not_updated, should_stop