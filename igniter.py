from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *

def run(epochs, model, criterion, optimizer, scheduler,
        train_loader, val_loader, device):
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    train_evaluator = create_supervised_evaluator(
        model,
        metrics={'accuracy': Accuracy(), 'nll': Loss(criterion)},
        device=device
    )
    validation_evaluator = create_supervised_evaluator(
        model,
        metrics={'accuracy': Accuracy(), 'nll': Loss(criterion)},
        device=device
    )
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names='all')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        scheduler.step()
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        pbar.log_message(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll)
)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        validation_evaluator.run(val_loader)
        metrics = validation_evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        pbar.log_message(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))
        pbar.n = pbar.last_print_n = 0
        
    tb_logger = TensorboardLogger(log_dir="log/")
    tb_logger.attach(trainer, 
                     log_handler=OutputHandler(tag="training", 
                                               output_transform=lambda loss:{"batchloss":loss},
                                               metric_names="all"),
                     event_name = Events.ITERATION_COMPLETED(every=100),
                     )
    tb_logger.attach(train_evaluator,
                     log_handler=OutputHandler(tag="training",
                                               metric_names=["nll", "accuracy"]),
                     event_name = Events.EPOCH_COMPLETED,
                     )
    tb_logger.attach(validation_evaluator,
                     log_handler=OutputHandler(tag="validation",
                                               metric_names=["nll", "accuracy"]),
                     event_name = Events.EPOCH_COMPLETED,
                     )
    tb_logger.attach(
            trainer,
            log_handler=OptimizerParamsHandler(optimizer),
            event_name = Events.ITERATION_COMPLETED(every=100)
            )
    tb_logger.attach(trainer, 
                     log_handler=WeightsScalarHandler(model),
                     event_name=Events.EPOCH_COMPLETED(every=100)
                     )
    tb_logger.attach(trainer, 
                     log_handler=WeightsHistHandler(model),
                     event_name=Events.EPOCH_COMPLETED(every=100)
                     )
    tb_logger.attach(trainer,
                     log_handler=GradsScalarHandler(model),
                     event_name=Events.ITERATION_COMPLETED(every=100))

    tb_logger.attach(trainer,
                     log_handler=GradsHistHandler(model),
                     event_name=Events.ITERATION_COMPLETED(every=100))

    # kick everything off

    trainer.run(train_loader, max_epochs=epochs)
    tb_logger.close()

    trainer.run(train_loader, max_epochs=epochs)