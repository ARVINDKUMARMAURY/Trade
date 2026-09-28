import os
import asyncio
import joblib
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile
from dotenv import load_dotenv  # <-- Naya addition

# ================= 1. CONFIGURATION & SETUP =================
# .env file se keys load karna (Local testing ke liye)
load_dotenv()  # <-- Yeh line .env file ko read karti hai

# Ab direct environment variables se keys fetch hongi
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_TOKEN')
API_KEY = os.getenv('BINANCE_API')
SECRET_KEY = os.getenv('BINANCE_SECRET')
TELEGRAM_CHAT_ID = os.getenv('CHAT_ID') 

# Security check: Agar koi key miss ho gayi toh script yahi ruk jayegi
if not TELEGRAM_BOT_TOKEN or not API_KEY or not SECRET_KEY:
    raise ValueError("⚠️ Keys missing hain! Kripya apni .env file ya Railway variables check karein.")

SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
TRADE_AMOUNT = 0.001

# Bot & Exchange Initialization
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'enableRateLimit': True,
})
# Sandbox (Testnet) mode ON rakha hai taaki real loss na ho. Live ke liye isey False karein.
exchange.set_sandbox_mode(True)

# State Variables
is_trading = False
trade_history = []

# ================= 2. MACHINE LEARNING MODEL LOAD =================
print("Loading ML Model...")
try:
    # Ensure karein ki aapne 'crypto_ml_model.pkl' train karke same folder mein rakhi hai
    model = joblib.load('crypto_ml_model.pkl')
    print("✅ ML Model Loaded Successfully!")
except Exception as e:
    print(f"⚠️ Warning: Model load nahi hua. Error: {e}")
    model = None


# ================= 3. TRADING LOGIC =================
async def fetch_and_predict():
    try:
        # Market data fetch karein (100 candles chahiye taaki SMA 50 ban sake)
        ohlcv = await exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Technical Indicators (Features) calculate karein
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df.dropna(inplace=True) # NaN values hata dein
        
        # Sabse latest (current) candle ka data nikaalein
        latest_data = df.iloc[-1:]
        current_price = latest_data['close'].values[0]
        
        # Model ke liye same wahi columns dein jo training mein use hue the
        features = latest_data[['close', 'volume', 'rsi', 'sma_20', 'sma_50']]
        
        if model is not None:
            prediction = model.predict(features)[0] # 1 for UP, 0 for DOWN
            confidence = model.predict_proba(features)[0]
        else:
            prediction = 0
            confidence = [0, 0]

        print(f"[{SYMBOL}] Price: ${current_price} | Prediction: {'UP' if prediction == 1 else 'DOWN'}")

        # Agar model kahta hai UP jayega (Buy signal)
        if prediction == 1 and confidence[1] > 0.55: # 55% se zyada confidence ho tabhi
            order = await exchange.create_market_buy_order(SYMBOL, TRADE_AMOUNT)
            
            # Record Store karein
            trade_data = {
                'Time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Symbol': SYMBOL,
                'Action': 'BUY',
                'Price': current_price,
                'Confidence': round(confidence[1] * 100, 2)
            }
            trade_history.append(trade_data)
            
            # Telegram par alert bhejein
            msg = f"🟢 **AI BUY Signal Executed**\nSymbol: {SYMBOL}\nPrice: ${current_price}\nConfidence: {trade_data['Confidence']}%"
            await bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown")

    except Exception as e:
        print(f"Error in trading loop: {e}")

async def trading_loop():
    global is_trading
    while is_trading:
        await fetch_and_predict()
        # 1 hour timeframe hai, par check har 5 minute mein karein
        await asyncio.sleep(300) 


# ================= 4. TELEGRAM COMMANDS =================

@dp.message(Command("start_bot"))
async def start_trading(message: Message):
    global is_trading
    if not is_trading:
        is_trading = True
        await message.answer("✅ **AI Trading Bot Started!**\nMarket analysis chal rahi hai.", parse_mode="Markdown")
        asyncio.create_task(trading_loop())
    else:
        await message.answer("⚠️ Bot already running hai.")

@dp.message(Command("stop_bot"))
async def stop_trading(message: Message):
    global is_trading
    is_trading = False
    await message.answer("🛑 Trading bot stopped.")

@dp.message(Command("status"))
async def check_status(message: Message):
    try:
        ticker = await exchange.fetch_ticker(SYMBOL)
        await message.answer(f"📊 **Current Status**\nSymbol: {SYMBOL}\nPrice: ${ticker['last']}\nBot Running: {is_trading}\nTotal Trades in Memory: {len(trade_history)}", parse_mode="Markdown")
    except Exception as e:
        await message.answer("⚠️ Exchange se connect nahi ho pa raha.")

# --- Database Management (Export) ---
@dp.message(Command("get_data"))
async def command_get_data(message: Message):
    if len(trade_history) == 0:
        await message.answer("⚠️ Abhi tak koi trade record nahi hua hai.")
        return
        
    df = pd.DataFrame(trade_history)
    csv_filename = "trading_data.csv"
    df.to_csv(csv_filename, index=False)
    
    document = FSInputFile(csv_filename)
    await bot.send_document(
        chat_id=message.chat.id, 
        document=document, 
        caption="📊 Aapka Trading Data Export!\n(Is file ko reply karke `/restore` command dene se data wapas load ho jayega)"
    )
    
    if os.path.exists(csv_filename):
        os.remove(csv_filename)

# --- Database Management (Restore) ---
@dp.message(Command("restore"))
async def restore_data_from_telegram(message: Message):
    global trade_history
    
    if not message.reply_to_message or not message.reply_to_message.document:
        await message.answer("⚠️ Kripya valid CSV file par reply karke `/restore` command bhejein.")
        return
        
    document = message.reply_to_message.document
    if not document.file_name.endswith('.csv'):
        await message.answer("⚠️ Ye file CSV format mein nahi hai.")
        return

    temp_filename = "restored_data.csv"
    
    try:
        await message.answer("⏳ Data restore kiya ja raha hai...")
        await bot.download(document, destination=temp_filename)
        
        df = pd.read_csv(temp_filename)
        trade_history = df.to_dict('records')
        
        await message.answer(f"✅ **Data Successfully Restored!**\nTotal {len(trade_history)} trades memory mein load ho gaye hain.", parse_mode="Markdown")
    except Exception as e:
        await message.answer(f"❌ File restore karne mein error aayi:\n`{e}`", parse_mode="Markdown")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# ================= 5. MAIN RUNNER =================
async def main():
    print("Starting AI Telegram Bot...")
    try:
        await dp.start_polling(bot)
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
    
