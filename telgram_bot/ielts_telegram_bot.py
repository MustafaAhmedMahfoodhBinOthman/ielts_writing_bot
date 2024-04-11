import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ConversationHandler, ContextTypes, MessageHandler, filters
from telegram.constants import ParseMode
import anthropic
import google.generativeai as genai
import random
import re
import time
from essay_evaluator import EssayEvaluator
import asyncio
import os
from server import server
from dotenv import load_dotenv

load_dotenv()
# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Define conversation states
TASK, QUESTION, ESSAY, TASK_TYPE_SPECIFICATION = range(4)

# Initialize the EssayEvaluator



Claude_API_KEY = os.getenv('Claude_API_KEY')
Gemini_API_Key = os.getenv('Gemini_API_Key') #mustafabinothman22
Gemini_API_Key2 = os.getenv('Gemini_API_Key2') #mustafanotion
Gemini_API_Key3 = os.getenv('Gemini_API_Key3') #mustafabinothman2003
Gemini_API_Key4 = os.getenv('Gemini_API_Key4') #mustafabinothman2023
Gemini_API_Key5 = os.getenv('Gemini_API_Key5') #www.binothman24
Bot_API_Token = os.getenv('Bot_API')



model = genai.GenerativeModel('gemini-1.0-pro-latest')
model_vision = genai.GenerativeModel('gemini-pro-vision')

keys = [Gemini_API_Key, Gemini_API_Key2, Gemini_API_Key3, Gemini_API_Key4, Gemini_API_Key5]
used_key = random.choice(keys)
opus = "claude-3-opus-20240229"
sonnet = "claude-3-sonnet-20240229"
haiku = "claude-3-haiku-20240307"
evaluator = EssayEvaluator(model, model_vision, keys, Claude_API_KEY)

# async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) :
#     await update.message.reply_text("Welcome to the IELTS Writing Evaluation Bot!")
#     await update.message.reply_text("Please choose the task type (Task 1 or Task 2):")
#     return TASK

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to the IELTS Writing Evaluation Bot!")
    keyboard = [
        [InlineKeyboardButton("Task 1", callback_data="Task 1")],
        [InlineKeyboardButton("Task 2", callback_data="Task 2")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Please choose the task type:", reply_markup=reply_markup)
    return TASK


async def task(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    task_type = query.data
    context.user_data['task'] = task_type

    if task_type == 'Task 1':
        keyboard = [
            [InlineKeyboardButton("Academic", callback_data="Academic")],
            [InlineKeyboardButton("General", callback_data="General")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(f"You have selected {task_type}. Please specify if the task is academic or general:", reply_markup=reply_markup)
        return TASK_TYPE_SPECIFICATION
    else:
        await query.edit_message_text(f"You have selected {task_type}.")
        await query.message.reply_text("Please provide the question:")
        return QUESTION

async def task_type_specification(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    task_type_specification = query.data
    context.user_data['task_type_specification'] = task_type_specification
    await query.edit_message_text(f"You have selected {context.user_data['task']} - {task_type_specification}.")
    await query.message.reply_text("Please provide the question:")
    return QUESTION
async def question(update: Update, context: ContextTypes.DEFAULT_TYPE) :
    question_text = update.message.text
    context.user_data['question'] = question_text
    await update.message.reply_text("Please write your essay:")
    return ESSAY

def check_word_count(essay_text: str, question_text: str, task_type: str) -> str:
    num_words = len(essay_text.split())
    q_words = len(question_text.split())

    if q_words == 0:
        return "Please provide the question."
    elif num_words == 0:
        return "Please write your essay."
    elif task_type.lower() == 'task 1' and num_words < 150:
        return "Your essay is too short. The written words count is {}. Please send the essay again with at least 150 words for Task 1. [Click here to start over](/start)".format(num_words)
    elif task_type.lower() == 'task 2' and num_words < 250:
        return "Your essay is too short. The written words count is {}. Please send the essay again with at least 250 words for Task 2. [Click here to start over](/start)".format(num_words)
    else:
        return None
    


async def essay(update: Update, context: ContextTypes.DEFAULT_TYPE) :
    essay_text = update.message.text
    print("Received essay from user")
    # print("----------------------------------")
    # print(essay_text)
    task = context.user_data['task']
    # print("----------------------------------")
    # print(task)
    task_type_specification = context.user_data.get('task_type_specification', '')
    # print("----------------------------------")
    # print(task_type_specification)
    question = context.user_data['question']
    # print("----------------------------------")
    # print(question)
    # Perform word count check
    word_count_result = check_word_count(essay_text, question, task)
    # print("----------------------------------")
    # print(word_count_result)

    # Perform word count check
    word_count_result = check_word_count(essay_text, question, task)
    if word_count_result is not None:
        await update.message.reply_text(word_count_result)
        return ESSAY

    message = await update.message.reply_text("Evaluating your essay. Please wait...")

    try:
        evaluation_task = asyncio.create_task(asyncio.to_thread(evaluator.evaluate_essay, essay_text, task, question, task_type_specification))

        for i in range(100, 0, -1):
            await message.edit_text(f"Evaluating your essay. it takes time please wait...\n\nEvaluation will be ready in {i} seconds...")
            await asyncio.sleep(1)

        evaluation_results = await evaluation_task

        await message.delete()

        await update.message.reply_text("IELTS Writing Evaluation Report\n", parse_mode='Markdown')

        await update.message.reply_text(evaluation_results["task"], parse_mode='Markdown')

        await update.message.reply_text(f"Number of Words in the essay: {evaluation_results['num_words']}\n", parse_mode='Markdown')

        await update.message.reply_text(f"Task Response: \nscore: {evaluation_results['task_response_score']}\n\n{evaluation_results['task_response_text']}\n", parse_mode='Markdown')

        await update.message.reply_text(f"Coherence and Cohesion: \nscore: {evaluation_results['coherence_cohesion_score']}\n\n{evaluation_results['coherence_cohesion_text']}\n", parse_mode='Markdown')

        await update.message.reply_text(f"Lexical Resources: \nscore: {evaluation_results['lexical_resources_score']}\n\n{evaluation_results['lexical_resources_text']}\n", parse_mode='Markdown')

        await update.message.reply_text(f"Grammar and Accuracy: \nscore: {evaluation_results['grammar_accuracy_score']}\n\n{evaluation_results['grammar_accuracy_text']}\n", parse_mode='Markdown')

        await update.message.reply_text(f"Overall Band Score:   {evaluation_results['overall_score']}", parse_mode='Markdown')
 
        await update.message.reply_text("**If you want a detailed feedback with more features**, \n\nTry the website for free\n https://ielts-writing-ai.streamlit.app/", parse_mode='Markdown')

        await update.message.reply_text("You can start evaluating again by clicking /start", parse_mode='Markdown')
        print("Evaluation completed and sent to user")

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        await update.message.reply_text("An error occurred while evaluating your essay. Please try again later. **You can start again by clicking /start.**", parse_mode='Markdown')
        return ESSAY

    return ConversationHandler.END
    

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Evaluation has stopped. Thank you for using the IELTS Writing Evaluation Bot!\n [Click here to start over](/start)")
    return ConversationHandler.END

def main() :
    print("Bot is starting...")
    application = Application.builder().token(Bot_API_Token).build()


    conv_handler = ConversationHandler(
    entry_points=[CommandHandler('start', start)],
    states={
        TASK: [CallbackQueryHandler(task)],
        TASK_TYPE_SPECIFICATION: [CallbackQueryHandler(task_type_specification)],
        QUESTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, question), CommandHandler('start', start)],
        ESSAY: [MessageHandler(filters.TEXT & ~filters.COMMAND, essay), CommandHandler('start', start)]
    },
    fallbacks=[CommandHandler('cancel', cancel)]
)

    application.add_handler(conv_handler)

    print("Bot is running and waiting for messages...")
    application.run_polling()

if __name__ == '__main__':
    
    server()
    main()
