import tkinter as tk

import requests

class ChatbotUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("Neurotic Tachyons")
        self.geometry("400x500")

        # Create widgets
        self.header_label = tk.Label(self, text="Neurotic Tachyons", font=("Arial", 14, "bold"))
        self.chat_history_text = tk.Text(self, state="disabled", wrap=tk.WORD)
        self.input_box = tk.Entry(self)
        self.send_button = tk.Button(self, text="Send", command=self.send_input)
        self.clear_button = tk.Button(self, text="Clear", command=self.clear_chat_history)

        # Place widgets
        self.header_label.pack(pady=10)
        self.chat_history_text.pack(fill=tk.BOTH, expand=True)
        self.input_box.pack(fill=tk.X, padx=10, pady=5)
        self.send_button.pack(side=tk.RIGHT, padx=5, pady=5)
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Bind input box to enter key press
        self.input_box.bind("<Return>", lambda event: self.send_input())

    def send_input(self):
        user_input = self.input_box.get()
        if user_input:
            self.chat_history_text.configure(state="normal")
            self.chat_history_text.insert(tk.END, f"User: {user_input}\n")
            self.chat_history_text.insert(tk.END, f"Neurotic Tachyons: {process_input(user_input)}\n")  # Replace with your chatbot's response logic
            self.chat_history_text.configure(state="disabled")
            self.input_box.delete(0, tk.END)

    def clear_chat_history(self):
        self.chat_history_text.configure(state="normal")
        self.chat_history_text.delete(1.0, tk.END)
        self.chat_history_text.configure(state="disabled")

# Placeholder function for processing input and generating response
def process_input(user_input):
    # Replace this with your chatbot's actual logic
    data = {"question": user_input}  # Create a dictionary for the data
    response = requests.post("http://localhost:6002/complete", json=data)
    # return "I'm still under development. Please try asking me something else."
    return response.text

if __name__ == "__main__":
    chatbot_ui = ChatbotUI()
    chatbot_ui.mainloop()