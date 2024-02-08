from torch.utils.tensorboard import SummaryWriter

from utils.globals import get_global_variable


class Logger:
    def __init__(
        self,
        log_dir=get_global_variable("log_dir"),
        group: str = None,
        text_logs: bool = False,
        log_interval: int = 1,
        step_bias: int = 0,
        reduction: str = "last",
        title: tuple[str, str] = ("iteration", "value"),
    ):
        """
        :param log_interval: in steps
        """

        self.log_dir = log_dir
        self.group = group
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)
        self.text_logs = text_logs
        self.log_interval = log_interval
        self.step_bias = step_bias
        self.reduction = reduction
        self.title = title
        self.current_step: int = 0
        self.current_values: list[float] = []

    def reset(self) -> None:
        match self.reduction:
            case "last":
                value = self.current_values[-1]
            case "sum":
                value = sum(self.current_values)
            case "mean":
                value = sum(self.current_values) / len(self.current_values)
            case _:
                raise NotImplementedError

        self.tb_writer.add_scalar(self.group, value, self.current_step + self.step_bias)

        if self.text_logs:
            print(
                self.title[0] + ": " + str(self.current_step + self.step_bias), end=", "
            )
            print(self.title[1] + ": " + str(value))

        self.current_values = []
        self.tb_writer.flush()

    def step(self, value: float) -> float:
        """
        Returns given value.
        """

        self.current_step += 1
        self.current_values.append(value)

        if self.current_step % self.log_interval == 0:
            self.reset()

        return value

    def add_step_bias(self, value: int) -> None:
        self.step_bias += value

    def __del__(self):
        self.tb_writer.close()
