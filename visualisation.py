import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# TODO: add legend somehow for return+shaped return plot

class Visualisation():
    """
    Class for making the plots.
    """
    
    def __init__(self, data_path, stage="Training", algorithm="Tabular-QL", shaping=False):
        self.data = pd.read_csv(data_path)
        self.stage = stage
        self.algorithm = algorithm    
        #self.shaping_alpha = data_path[-7:-4]
        self.shaping = shaping
    

    def actions2Strings(self):
        if self.data["Action"].dtype != "str":
            self.data["Action"] = self.data["Action"].replace([0, 1, 2], ["Sell", "Hold", "Buy"])
        if self.data["Taken Action"].dtype != "str":
            self.data["Taken Action"] = self.data["Taken Action"].replace([0, 1, 2], ["Sell", "Hold", "Buy"])
    

    def get_realActions(self):
        if "Executed Action" not in self.data.columns:
            self.data["Executed Action"] = self.data["Action"]
            self.data["Executed Action"] = self.data["Executed Action"].mask(self.data["Executed Action"] != self.data["Taken Action"], self.data["Taken Action"] + " instead of " + self.data["Executed Action"])


    def add_return(self):
        """
        Adds the return to the data by taking the cumulative sum of the reward.
        """
        self.data["Return"] = self.data["Reward"].cumsum()


    def add_shaped_return(self):
        """
        Adds the shaped return to the data by taking the cumulative sum of the shaped reward.
        """
        self.data["Shaped Return"] = self.data["Shaped Reward"].cumsum()


    def action_by_price(self, start=0, end=-1, save=False):
        """
        Plots and saves a figure displaying the price and taken actions each hour for a specified time frame.

        :param start: start of time frame
        :type start: int, optional, defaults to 0
        :param end: end of time frame
        :type end: int, optional
        :param save: whether to save the figure or not, defaults to False
        :type save: bool, optional
        """

        # Prepare action column for plotting
        self.actions2Strings()
        self.get_realActions()


        # Make overlapping plots
        sns.set(palette=("#1fa1c1", '#E36414', '#FB9637', '#9A031E', '#5F0F40'))
        line_plot = sns.lineplot(data=self.data[start:end], x=self.data.index[start:end], y="Price")
        sns.set(palette=('#8CBA80', '#FB9637', '#9A031E', "#1A3175", "#B45578"))
        scatter_plot = sns.scatterplot(data=self.data[start:end], x=self.data.index[start:end], y="Price", hue="Executed Action")
        
        # Label axes and title
        plt.xlabel("Hours")
        plt.ylabel("Price (€)")
        plt.title(f"{self.algorithm}-{self.stage}: Price & Taken Action Each Hour", fontweight="bold")

        # Save and show figure
        if save == True:
            plt.savefig(f"plots/{self.algorithm}_{self.stage}_Shaping={self.shaping}_ActionByPrice.png")
        plt.show()
        

    def action_by_waterlevel(self, start=0, end=-1, save=False):
        """
        Plots and saves a figure displaying the waterlevel and taken actions each hour for a specified time frame.

        :param start: start of time frame
        :type start: int, optional, defaults to 0
        :param end: end of time frame
        :type end: int, optional
        :param save: whether to save the figure or not, defaults to False
        :type save: bool, optional
        """

        # Prepare action column for plotting
        self.actions2Strings()
        self.get_realActions()
        
        # Make overlapping plots and filling
        sns.set(palette=("#1fa1c1", '#E36414', '#FB9637', '#9A031E', '#5F0F40'))
        line_plot = sns.lineplot(data=self.data[start:end], x=self.data.index[start:end], y= "Waterlevel")
        plt.fill_between(self.data.index[start:end].values, self.data["Waterlevel"][start:end].values, alpha = 0.3)
        sns.set(palette=('#8CBA80', '#FB9637', '#9A031E', "#1A3175", "#B45578"))
        scatter_plot = sns.scatterplot(data=self.data[start:end], x=self.data.index[start:end], y="Waterlevel", hue="Executed Action")
        
        # Label axes and title
        plt.xlabel("Hours")
        plt.ylabel("Waterlevel ($m^3$)")
        plt.title(f"{self.algorithm}-{self.stage}: Waterlevel & Taken Action Each Hour", fontweight="bold")

        # Save and show figure
        if save == True:
            plt.savefig(f"plots/{self.algorithm}_{self.stage}_Shaping={self.shaping}_ActionByWaterlevel.png")
        plt.show()


    def action_by_return(self, start=0, end=-1, save=False):
        """
        Plots and saves a figure displaying the return and taken actions each hour for a specified time frame.

        :param start: start of time frame
        :type start: int, optional, defaults to 0
        :param end: end of time frame
        :type end: int, optional
        :param save: whether to save the figure or not, defaults to False
        :type save: bool, optional
        """

        # Prepare action column for plotting
        self.actions2Strings()
        self.get_realActions()

        # Add return column to data
        self.add_return()

        # Make overlapping plots
        sns.set(palette=("#1fa1c1", '#E36414', '#FB9637', '#9A031E', '#5F0F40'))
        line_plot = sns.lineplot(data=self.data[start:end], x=self.data.index[start:end], y="Return")
        sns.set(palette=('#8CBA80', '#FB9637', '#9A031E', "#1A3175", "#B45578"))
        scatter_plot = sns.scatterplot(data=self.data[start:end], x=self.data.index[start:end], y="Return", hue="Executed Action")
        
        # Label axes and title
        plt.xlabel("Hours")
        plt.ylabel("Return (€)")
        plt.title(f"{self.algorithm}-{self.stage}: Return & Taken Action Each Hour", fontweight="bold")
        
        # Save and show figure
        if save == True:
            plt.savefig(f"plots/{self.algorithm}_{self.stage}_Shaping={self.shaping}_ActionByReturn.png")
        plt.show()


    def action_by_bothReturns(self, start=0, end=-1, save=False):
        """""
        Plots and saves a figure displaying the price and taken actions each hour for a specified time frame.

        :param start: start of time frame
        :type start: int, optional, defaults to 0
        :param end: end of time frame
        :type end: int, optional
        :param save: whether to save the figure or not, defaults to False
        :type save: bool, optional
        """""

        # Prepare action column for plotting
        self.actions2Strings()
        self.get_realActions()

        # Add return column to data
        self.add_return()
        self.add_shaped_return()

        # Make overlapping plots
        sns.set(palette=("#1fa1c1", '#E36414', '#FB9637', '#9A031E', '#5F0F40'))
        line_plot1 = sns.lineplot(data=self.data[start:end], x=self.data.index[start:end], y="Return")
        line_plot2 = sns.lineplot(data=self.data[start:end], x=self.data.index[start:end], y="Shaped Return")
        sns.set(palette=('#8CBA80', '#FB9637', '#9A031E', "#1A3175", "#B45578"))
        scatter_plot = sns.scatterplot(data=self.data[start:end], x=self.data.index[start:end], y="Return", hue="Executed Action")
        
        # Label axes and title
        plt.xlabel("Hours")
        plt.ylabel("Euros")
        plt.title(f"{self.algorithm}-{self.stage}: Return, Shaped Return & Taken Action Each Hour", fontweight="bold")
        
        # Save and show figure
        if save == True:
            plt.savefig(f"plots/{self.algorithm}_{self.stage}_Shaping={self.shaping}_ActionByBothReturns.png")
        plt.show()


    def policy_2_3D(self):
        
        arr = self.data.to_numpy()
    
        fig = go.Figure(data=[go.Surface(z=arr, x=self.data["Waterlevel"], y=self.data["Price"])])

        fig.update_layout(title=f'Maximum Q-Values Across State Space', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
        fig.show()


    def train_and_val_curve(self):
        
        # Make line plot
        fig = sns.lineplot(data=self.data, x="Episode", y="Return", hue="Stage")

        # Label axes and title
        plt.xlabel("Episode")
        plt.ylabel("Episode Return")
        plt.title(f"{self.algorithm}: Return for Training and Validation", fontweight="bold")
        plt.show()

    def training_curve(self):
        # Make line plot
        fig = sns.lineplot(x = self.data["Episode"].where(self.data["Stage"]=="Training") , y=self.data["Return"].where(self.data["Stage"]=="Training"))

        # Label axes and title
        plt.xlabel("Episode")
        plt.ylabel("Episode Return")
        plt.title(f"{self.algorithm}: Return for Training and Validation", fontweight="bold")
        plt.show()
