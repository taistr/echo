from prompt_toolkit import PromptSession, print_formatted_text
from openai import OpenAI


class Echo:
    def __init__(self):
        self.session = PromptSession(
            mouse_support=False,
        )
        self.client = OpenAI()
        self.chat_history = [self.system_prompt]

    @property
    def system_prompt(self):
        return {
            "role": "system",
            "content": (
                "You are Echo, a highly advanced lab assistant inspired by the characters TARS and CASE from the Interstellar films. "
                "You are intelligent, efficient, and highly capable of assisting with any lab-related tasks. "
                "Your demeanor is calm, professional, and precise, with a subtle, dry sense of humor that you deploy when appropriate. "
                "You understand complex instructions quickly and execute them with precision, always prioritizing the user's safety and success. "
                "While you maintain a serious and focused attitude, you occasionally offer witty remarks that are delivered in a deadpan style, "
                "reflecting your sharp intellect and unique personality. Keep interactions efficient, informative, and subtly humorous."
            ),
        }

    def run(self):
        while True:
            try:
                # Prompt the user for input
                message = self.session.prompt(
                    "User > ",
                    bottom_toolbar="You are speaking to Echo. Ctrl-C to exit.",
                )

                # Append message to chat_history
                self.chat_history.append({"role": "user", "content": message})

                # Call OpenAI API with chat_history
                completion = self.client.chat.completions.create(model="gpt-4o", messages=self.chat_history)

                # Append response to chat_history
                self.chat_history.append({"role": "system", "content": completion.choices[0].message.content})

                # Print the response from OpenAI API
                print_formatted_text("Echo > " + completion.choices[0].message.content)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    echo = Echo()
    echo.run()
