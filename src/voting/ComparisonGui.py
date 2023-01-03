import pandas as pd
import tkinter as tk
from tkinter import messagebox
import time

class ComparisonGui():
    def __init__(self, df_movies):
        self.df_movies = df_movies
        self.n = df_movies.shape[0]
    
        self.root = tk.Tk()
        #self.root.lift()
        #self.root.attributes('-topmost', True)
        self.root.title('Movie comparison')

        # 1. Header, 2. main window with comparison, 3. footer with submission button
        # Order them in priority so we always have a submit button and header if space runs out
        self.footer = tk.Frame(self.root, width=250, bg="grey")
        self.footer.pack(side="bottom", fill="both", expand=True)

        self.header = tk.Frame(self.root, width=250)
        self.header.pack(side="top", fill="both", expand=True)

        self.main = tk.Frame(self.root, width=250)
        self.main.pack(side="top", fill="both", expand=True)


        # Header: title label
        self.title = tk.Label(
            self.header, 
            text="Choose the winning movie:", 
            font="Arial 14", 
            foreground="white", 
            bg="grey"
        )
        self.title.pack(side="top", fill="both", expand=True)

        # Main: comparison panel
        self.comp = tk.Frame(self.main, width=250, bg="grey")
        self.comp.grid(sticky="nsew")

        self.main.grid_rowconfigure(0, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        self.result_dict = {
            i: {
                'choice': tk.IntVar(), 
                'timestamp': tk.IntVar()
            } for i in range(0, self.n)
        }

        
        for i in range(0, self.n):
            self.comp.grid_rowconfigure(i, weight=1)
            self.comp.grid_columnconfigure(0, weight=1)
            self.comp.grid_columnconfigure(1, weight=1)

            self.create_button(
                text=self.df_movies.loc[i, "title_1"], 
                var=self.result_dict[i]['choice'], 
                value=1, 
                row=i, 
                column=0
            )

            self.create_button(
                text=self.df_movies.loc[i, "title_2"], 
                var=self.result_dict[i]['choice'], 
                value=2, 
                row=i, 
                column=1
            )

        # Footer: submission button
        self.submit = tk.Button(
            self.footer, 
            text="Submit", 
            font="Arial 14", 
            command=lambda: self.set_submissions()
        )
        self.submit.pack(side="top", padx=2, pady=2, fill="both", expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.closing_message)
        self.root.mainloop()


    def create_button(self, text, var, value, row, column):
        self.button = tk.Radiobutton(
            self.comp, 
            indicatoron=False, 
            font="Arial 14",
            selectcolor="green"
        )
        self.button.configure(
            text=text, 
            variable=var,
            value=value,
            command=lambda: self.save_timestamp(row)
        )
        self.button.grid(
            row=row, 
            column=column, 
            sticky="nsew", 
            padx=(2, 2), 
            pady=(2, 2)
        )


    def save_timestamp(self, idx):
        self.result_dict[idx]['timestamp'].set(int(time.time()))


    def set_submissions(self):
        __comps = [
            (
                key,
                value['choice'].get(), 
                value['timestamp'].get()
            ) 
            for (key, value) in self.result_dict.items()
        ]
        __df_votes = pd.DataFrame(__comps, columns=['idx', 'choice', 'timestamp'])
        self.df_results = pd.merge(
            self.df_movies[['tconst_1', 'tconst_2']], 
            __df_votes[['choice', 'timestamp']], 
            left_index=True, 
            right_index=True
        )
        self.df_results = self.df_results[['timestamp', 'tconst_1', 'tconst_2', 'choice']]
        self.max_timestamp = max(self.df_results['timestamp'])
        self.root.destroy()

    def closing_message(self):
        if messagebox.askokcancel("Quit", "Do you want to stop casting votes?"):
            self.root.destroy()
    
