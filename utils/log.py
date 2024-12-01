import logging
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, log_exp, log_dir="experiments", log_file="log.txt"):
        self.log_dir = log_dir
        self.log_exp = log_exp
        self.log_file = log_file

        self.log_path = os.path.join(self.log_dir, self.log_exp, self.log_file)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            # format="%(asctime)s - %(levelname)s : \n%(message)s",
            format="%(message)s",
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler(sys.stdout) # Print to console as well
            ]
        )
        self.logger = logging.getLogger()

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)


# def save_plot(fig, plot_dir, plot_exp, plot_file_name):
#     if plot_file_name is None:
#         plot_file_name = "plot" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
#
#     plot_path = os.path.join(plot_dir, plot_exp, plot_file_name)
#     fig.savefig(plot_path)
#     plt.close(fig)
#     print(f"Plot saved at {plot_path}")

