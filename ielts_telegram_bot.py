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
from groq import Groq
import re
from supabase import create_client, Client
# from server import server
from dotenv import load_dotenv
# from datetime import datetime
import datetime
import time
load_dotenv()
# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Define conversation states
TASK, QUESTION, ESSAY, TASK_TYPE_SPECIFICATION, IMAGE_OR_SKIP, IMAGE_UPLOAD = range(6)

# Initialize the EssayEvaluator
# Claude_API_KEY = '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'
# Gemini_API_Key = 'AIzaSyAtnlV6rfm_OsSt9M_w9ZaiFn3NjdjSVuw'
# Gemini_API_Key2 = 'AIzaSyDbU_8cAQCAhr59bqtGf40FV-92KCKkLWs'
# Gemini_API_Key3 = 'AIzaSyBOb6xrGvLxRBvgMEUyWvTSGKZVDGT4j3w'
# Gemini_API_Key4 = 'AIzaSyB5Cy4KIg4xKwz2poq3sywJEvqI0BL10iQ'
# Gemini_API_Key5 = 'AIzaSyBUpws7IJIKo9rZI1YKSBPQj_RpPWwTqFo'

url = "https://twrfzriopjdkicchfqzs.supabase.co"
# url = os.getenv('url')
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR3cmZ6cmlvcGpka2ljY2hmcXpzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTY2MTQ0NTgsImV4cCI6MjAzMjE5MDQ1OH0.WW-BxGtjADTRMuX2lBkPvSy2cbjxzYZls67VlgQlRF0"
# key = os.getenv('supabase')
supabase: Client = create_client(url, key)
Claude_API_KEY = os.getenv('Claude_API_KEY')
Gemini_API_Key = os.getenv('Gemini_API_Key') #mustafabinothman22
Gemini_API_Key2 = os.getenv('Gemini_API_Key2') #mustafanotion
Gemini_API_Key3 = os.getenv('Gemini_API_Key3') #mustafabinothman2003
Gemini_API_Key4 = os.getenv('Gemini_API_Key4') #mustafabinothman2023
Gemini_API_Key5 = os.getenv('Gemini_API_Key5') #www.binothman24

groq_API1 = os.getenv('groq_API1')
groq_API2 = os.getenv('groq_API2') #mustafabinothman22
groq_API3 = os.getenv('groq_API3') #mustafanotion
groq_API4 = os.getenv('groq_API4') #mustafabinothman2003
groq_API5 = os.getenv('groq_API5') #mustafabinothman2023
groq_API6 = os.getenv('groq_API6') #www.binothman24
Bot_API_Token = os.getenv('Bot_API')
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')



model = genai.GenerativeModel('gemini-1.0-pro-latest')
model_vision = genai.GenerativeModel('gemini-pro-vision')
keys = [groq_API1, groq_API2, groq_API3, groq_API4, groq_API5]

# keys = [Gemini_API_Key, Gemini_API_Key2, Gemini_API_Key3, Gemini_API_Key4, Gemini_API_Key5]
used_key = random.choice(keys)
opus = "claude-3-opus-20240229"
sonnet = "claude-3-sonnet-20240229"
haiku = "claude-3-haiku-20240307"
evaluator = EssayEvaluator(model, model_vision, used_key, REPLICATE_API_TOKEN)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # user = update.message.from_user
    # user_id = user.id
    
    # username = user.username  # Retrieve the username

    # print("User ID:", user_id)
    # print("Username:", username if username else "No username available")

    await update.message.reply_text("Welcome to the IELTS Writing Evaluation Bot!")
    # await update_user_data(user_id, username)  # Pass the username to the update function

    keyboard = [
        [InlineKeyboardButton("Task 1", callback_data="Task 1")],
        [InlineKeyboardButton("Task 2", callback_data="Task 2")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Please choose the task type:", reply_markup=reply_markup)
    return TASK  # Ensure TASK is defined or replace with appropriate state management constant

async def update_user_data(user_id, username):
    import pytz
    utc_time = datetime.datetime.now()
    # Define the Mecca timezone
    mecca_tz = pytz.timezone('Asia/Riyadh')  # Riyadh is in the same time zone as Mecca
    # Convert UTC time to Mecca time
    mecca_time = utc_time.replace(tzinfo=pytz.utc).astimezone(mecca_tz)
    # Format the Mecca time as "Day/Month/Year Hour:Minute"
    formatted_time = mecca_time.strftime("%d/%m/%Y %H:%M")
    username = username if username is not None else "None"
    data = {
        "user_id": user_id,
        "last_used": formatted_time,
        "username": username
    }
    # Check if user exists
    try:
        user_exists = supabase.table("ielts_writing_bot_users").select("user_id").eq("user_id", user_id).execute()
        if user_exists.data:
            # Update existing user
            supabase.table("ielts_writing_bot_users").update(data).eq("user_id", user_id).execute()
        else:
            # Insert new user
            supabase.table("ielts_writing_bot_users").insert(data).execute()
        print("user id get updated: ", user_id,"username: ",username, "time: ", formatted_time)
    except Exception as e:
        print("error while updating the user data: ", e)

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

    if context.user_data['task'] == 'Task 1' and task_type_specification == 'Academic':
        keyboard = [
            [InlineKeyboardButton("Yes", callback_data="Yes")],
            [InlineKeyboardButton("No", callback_data="No")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text("Do you have an image of the chart?", reply_markup=reply_markup)
        return IMAGE_OR_SKIP
    else:
        await query.edit_message_text(f"You have selected {context.user_data['task']} - {task_type_specification}.")
        await query.message.reply_text("Please provide the question:")
        return QUESTION
async def question(update: Update, context: ContextTypes.DEFAULT_TYPE) :
    question_text = update.message.text
    context.user_data['question'] = question_text
    await update.message.reply_text("Please write your essay:")
    return ESSAY
async def image_or_skip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "Yes":
        await query.edit_message_text("Please send the image of the chart:")
        return IMAGE_UPLOAD
    else:
        await query.edit_message_text("You have chosen to skip the image upload.")
        await query.message.reply_text("Please provide the question:")
        return QUESTION

async def image_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.photo:
        photo_file = await update.message.photo[-1].get_file()
        file_path = photo_file.file_path
        file_extension = file_path.split('.')[-1].lower()
        print(file_extension)
        if file_extension in ['jpg', 'jpeg', 'png']:
            context.user_data['chart_image'] = file_path
            await update.message.reply_text("Image received. Please provide the question:")
            return QUESTION
        else:
            await update.message.reply_text("Please send a valid image format (JPG, JPEG, PNG).")
            return IMAGE_UPLOAD
    elif update.message.document:
        document = update.message.document
        file_extension = document.file_name.split('.')[-1].lower()
        print(file_extension)
        if file_extension in ['jpg', 'jpeg', 'png']:
            document_file = await document.get_file()
            context.user_data['chart_image'] = document_file.file_path
            await update.message.reply_text("Image received. Please provide the question:")
            return QUESTION
        else:
            await update.message.reply_text("Please send a valid image format (JPG, JPEG, PNG).")
            return IMAGE_UPLOAD
    else:
        await update.message.reply_text("Please send a valid image format (JPG, JPEG, PNG).")
        return IMAGE_UPLOAD
async def share_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    bot_username = (await context.bot.get_me()).username
    share_message = (
        f"Check out this amazing IELTS Writing Evaluation Bot! It provides detailed feedback on your essays just like an IELTS examiner. Try it out for Free: @{bot_username}"
    )
    await query.message.reply_text(share_message)
    await query.message.reply_text("To share this bot, please forward the above message .")
def check_word_count(essay_text: str, question_text: str, task_type: str) -> str:
    num_words = len(essay_text.split())
    q_words = len(question_text.split())

    if q_words < 5:
        return "Please provide the question."
    elif num_words == 0:
        return "Please write your essay."
    elif task_type.lower() == 'task 1' and num_words < 150:
        return "Your essay is too short. The written words count is {}. Please send the essay again with at least 150 words for Task 1. [Click here to start over](/start)".format(num_words)
    elif task_type.lower() == 'task 2' and num_words < 250:
        return "Your essay is too short. The written words count is {}. Please send the essay again with at least 250 words for Task 2. [Click here to start over](/start)".format(num_words)
    else:
        return None

def is_valid_word(word):
    # Check if the word is made up of English letters, numbers, or allowable punctuation
    if re.fullmatch(r'[a-zA-Z0-9.,?!;:\'"\(\)\[\]{}\-_%]+', word):
        return True
    return False

def contains_no_urls(text):
    # Check if the text contains URL patterns
    if re.search(r'\bhttps?://|www\.\b', text):
        return False
    return True

def check_essay(essay):
    # Normalize the text by replacing newlines with spaces
    normalized_essay = essay.replace('\n', ' ')

    # Check for URLs in the entire text first
    if not contains_no_urls(normalized_essay):
        return False

    # Split the essay into words considering punctuation
    words = re.findall(r'\b[\w.,?!;:\'"\(\)\[\]{}\-_%]+\b', normalized_essay)

    # Check each word
    for word in words:
        if not is_valid_word(word):
            print("The essay contains non-English words or links")
            return False
    return True

async def essay(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    user_id = user.id
    username = user.username  # Retrieve the username
    essay_text = update.message.text
    print("Received essay from user")

    task = context.user_data['task']
    task_type_specification = context.user_data.get('task_type_specification', '')
    question = context.user_data['question']

    # Perform essay validation
    if not check_essay(essay_text):
        await update.message.reply_text("Your essay contains invalid content (URLs or non-English words). Please send a valid essay.")
        return ESSAY

    # Perform word count check
    word_count_result = check_word_count(essay_text, question, task)
    if word_count_result is not None:
        await update.message.reply_text(word_count_result)
        return ESSAY
    num_words = len(essay_text.split())
    await update.message.reply_text(f"Number of Words in the essay: {num_words}")
    message = await update.message.reply_text("Evaluating your essay. Please wait...")

    try:
        evaluation_task = asyncio.create_task(asyncio.to_thread(evaluator.evaluate_essay, essay_text, task, question, task_type_specification))

        for i in range(70, 0, -1):
            if evaluation_task.done():
                break
            await message.edit_text(f"Evaluating your essay. It takes time, please wait...\n\nEvaluation will be ready in {i} seconds...")
            await asyncio.sleep(1)

        if not evaluation_task.done():
            await message.edit_text("Sorry for inconveniences Evaluation is taking longer than expected. Please wait a bit more..., if this happens frequintly please contact me @ielts_pathway\n\nآسف  قد يستغرق التقييم وقتًا أطول من المتوقع. يرجى الانتظار أكثر قليلاً...، إذا حدث هذا بشكل متكرر، فيرجى التواصل معي")

        # try:
            # evaluation_results, error_message = await evaluation_task

        #     if error_message:  # Check for error message
        #         print(f"An error occurred during evaluation: {error_message}")
        #         await update.message.reply_text(error_message, parse_mode='Markdown')
        #         return ESSAY
        # except Exception as e:
        #     print(f"An error occurred during evaluation: {e}")
        #     await update.message.reply_text("An error occurred while evaluating your essay. Please try again later. **You can start again by clicking /start.\n\nif the problem persists, please contact me @ielts_pathway", parse_mode='HTML')
        #     return ESSAY
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
        await update_user_data(user_id, username)  # Pass the username to the update function

        await update.message.reply_text("**Thank you for using IELTS Writing Evaluation Bot.\nIf you want a detailed feedback with more features, \n\nTry the website for free\n https://ielts-writing-ai.streamlit.app/**", parse_mode='Markdown')

        # await update.message.reply_text("You can start evaluating again by clicking /start\n\n if you find it helpful Do not forget sharing the bot 😊", parse_mode='Markdown')
        share_message = (
        f"Check out this amazing IELTS Writing Evaluation Bot! It provides detailed feedback on your essays just like an IELTS examiner. Try it out for Free: @ielts_writing2_bot"
    )
        keyboard = [
            [InlineKeyboardButton("Share the Bot", switch_inline_query=share_message)]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("You can start evaluating again by clicking /start\n\n if you find it helpful Do not forget sharing the bot 😊\n\n if you have any suggestions to improve the bot please contact me @ielts_pathway", reply_markup=reply_markup, parse_mode='HTML')

        # await update.message.reply_text("This bot is currently in Beta. If there is any issue or suggestion, please contact me @mustafa_binothman.", parse_mode='Markdown')
        # await update.message.reply_text("Do not forget sharing the bot 😊", parse_mode='Markdown')
        print("Evaluation completed and sent to user")

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        await update.message.reply_text("An error occurred while evaluating your essay. Please try again later. \n if the problem persists, please contact me  @ielts_pathway\nYou can start again by clicking /start.\n or you can evaluate your essay via our website\n https://ielts-writing-ai.streamlit.app/", parse_mode='HTML')
        await update.message.reply_text("حدث خطأ أثناء تقييم مقالتك. يرجى المحاولة مرة أخرى لاحقًا .\n إذا استمرت المشكلة، يرجى التواصل معي @ielts_pathway.", parse_mode='HTML')
        return ESSAY

    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text("Evaluation has stopped. Thank you for using the IELTS Writing Evaluation Bot!\n [Click here to start over](/start)")
    return ConversationHandler.END

def main():
    print("Bot is starting...")
    application = Application.builder().token(Bot_API_Token).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            TASK: [CallbackQueryHandler(task)],
            TASK_TYPE_SPECIFICATION: [CallbackQueryHandler(task_type_specification)],
            IMAGE_OR_SKIP: [CallbackQueryHandler(image_or_skip)],
            IMAGE_UPLOAD: [MessageHandler(filters.PHOTO | filters.Document.ALL, image_upload)],
            QUESTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, question), CommandHandler('start', start)],
            ESSAY: [MessageHandler(filters.TEXT & ~filters.COMMAND, essay), CommandHandler('start', start)]
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    application.add_handler(conv_handler)
    application.add_handler(CallbackQueryHandler(share_bot, pattern='share_bot'))

    print("Bot is running and waiting for messages...")
    application.run_polling()

if __name__ == '__main__':
    main()
