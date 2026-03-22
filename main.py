import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
load_dotenv()



def main():
    print("Hello from langchain-course!")
    print(os.environ.get("OPENAI_API_KEY"))
    information = """Ruhollah Mostafavi Musavi Khomeini[d] (17 May 1900[c] – 3 June 1989) was an Iranian political revolutionary and Shia cleric who served as the first supreme leader of Iran from 1979 until his death in 1989. He was the main leader of the 1979 Iranian Revolution, which overthrew Mohammad Reza Pahlavi and transformed Iran into an Islamic republic.

Born in the city of Khomeyn, in what is now Iran's Markazi province, Khomeini's father was murdered when he was two years old. He began studying the Quran and Arabic from a young age assisted by his relatives. Khomeini became a high ranking cleric in Twelver Shi'ism, an ayatollah, a marja' ("source of emulation"), a mujtahid, faqīha and a hafiz (an expert in fiqh), and author of more than 40 books. His opposition to the White Revolution resulted in his state-sponsored expulsion to Bursa in 1964. Nearly a year later, he moved to Najaf, where speeches he gave outlining his religiopolitical theory of Guardianship of the Jurist were compiled into the book Islamic Government.

After the success of the Iranian Revolution, Khomeini served as the country's de facto head of state from February 1979 until his appointment as supreme leader in December of that same year. Khomeini was Time magazine's Man of the Year in 1979 for his international influence and in the next decade was described as the "virtual face of Shia Islam in Western popular culture". He was known for his support of the hostage takers during the Iran hostage crisis; his fatwa calling for the murder of Indian-born British novelist Salman Rushdie for Rushdie's description of the Islamic prophet Muhammad in his novel The Satanic Verses, which Khomeini considered blasphemous; pursuing the overthrow of Saddam Hussein in the Iran–Iraq War; and for referring to the United States as the "Great Satan" and Israel as the "Little Satan".
"""
    
    summary_template = """Given the summary information {information} about a person, I want to create:
                          1. A short summary
                          2. Two interesting facts about them
                       """
    
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template)
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        #model="gemma-7b-it",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.9
    )
    # llm = ChatOllama(temperature=0,model='gemma3:270m')
    
    summary_chain = summary_prompt_template | llm
    
    response = summary_chain.invoke(input={"information": information})
    
    print(response.content)

if __name__ == "__main__":
    main()
