from datetime import datetime
import logging
import time
from email.message import EmailMessage
import smtplib
from numpy.core import numeric
from pandas import DataFrame as df
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp
import io
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
logger = logging.Logger(name = "Utils", level=logging.INFO)
class Utils(object):
    """
    docstring
    """
    # def __init__(self):
    #     pass
    
    @staticmethod
    def countdown(t: int):
        while t:
            mins, secs = divmod(t, 60)
            timeformat = '{:02d}:{:02d}'.format(mins, secs)
            logger.info(timeformat)
            time.sleep(1)
            t -= 1
        logger.info('Continue!')

    @staticmethod
    def fig_to_mail(plt):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        
        part = MIMEBase('application', "octet-stream")
        part.set_payload(buf.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        'attachment; filename="%s"' % 'stock_chart.png')
        plt.close("all")
        buf.truncate()
        return part

    @staticmethod
    def send_mm_mail(subject: str, body: str, plt=None):
        gmail_user = 'creaturives@gmail.com'
        gmail_password = 'ftskmtgcljgalzmc'

        sent_from = gmail_user
        to = 'tomas.kuttler@gmail.com'

        # msg = EmailMessage()
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = gmail_user
        msg['To'] = to
        # msg['encode'] = 'utf-8'
        msg.attach(MIMEText(body, 'plain'))

        # try:
        if plt:
            part = Utils.fig_to_mail(plt)
            msg.attach(part)
        
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        
        text = msg.as_string()
        server.sendmail(sent_from, to, text)
        # server.send_message(text)
        server.close()

  
        
    @staticmethod
    def send_mail(subject: str, body: str):
        gmail_user = 'creaturives@gmail.com'
        gmail_password = 'ftskmtgcljgalzmc'

        sent_from = gmail_user
        to = ['tomas.kuttler@gmail.com']

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = gmail_user
        msg['To'] = to
        # part1 = MIMEText(unicode('text'), "plain")
        # part2 = MIMEText('<html><body>HELLO</body></html>', "html")
        # msg.attach(part1)
        # msg.attach(part2)
        # plt.figure()
        # plt.plot([1, 2])
        # plt.title("test")
        # buf = io.BytesIO()
        # plt.savefig(buf, format='png')
        # buf.seek(0)
        # buf.close()
        # h = Header('encode', 'utf-8')
        # msg['Subject'] = h

        msg.set_content(body)

        # image = MIMEImage(buf.getvalue())
        # msg.attach(image)
        # need import image from PIL
        # im = Image.open(buf)
        # im.show()

        # text = MIMEText("test")
        # msg.attach(text)
        # image = MIMEImage(buf.getvalue())
        # image.add_header('11111', '<image1>')
        # msgAlternative.attach(image)
        # msg.attach(MIMEImage(buf.getvalue()))
        # msg.set_content(body)

        # try:

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        # server.sendmail(sent_from, to, msg.as_string())
        server.send_message(msg)
        # server.sendmail(sent_from, to,  email_text)
        server.close()

        #     logger.debug('Email sent!')
        # except:
        #     logger.debug('Something went wrong...')

        # def save_chart(self):
        #     plt.figure()
        #     plt.plot([1, 2])
        #     plt.title("test")
        #     buf = io.BytesIO()
        #     plt.savefig(buf, format='png')
        #     buf.seek(0)
        #     # need import image from PIL
        #     # im = Image.open(buf)
        #     # im.show()
        #     buf.close()

        # def countdown(self, t):
        #     while t:
        #         mins, secs = divmod(t, 60)
        #         timeformat = '{:02d}:{:02d}'.format(mins, secs)
        #         logger.debug(timeformat, end='\r')
        #         time.sleep(1)
        #         t -= 1
        #     logger.debug('Continue!\n\n\n\n\n')
        
    @staticmethod
    def convert_to_minutes(data: str) -> int:
        minutes = 0
        
        if not isinstance(data, str) and not isinstance(data, int):
            logger.error("ERROR: Data for convertion must be str or int \r\n")
            return None
            
        if isinstance(data, int):
            minutes = data
            
        elif isinstance(data, str) and data=="":
            minutes = 0
        
        elif data[-1].lower() == "h":
            minutes = int(data[0:len(data)-1]) * 60

        elif data[-1].lower() == "d":
            minutes = int(data[0:len(data)-1]) * 60 * 24

        elif data[-1].lower() == "m":
            minutes = int(data[0:len(data)-1])

        else:
            try:
                minutes = int(data)
            except:
                logger.error("Please add correct timeformat eg: 1m, 1h, 1d or 1")

        
        return minutes

    @staticmethod
    def calc_perc( first, last, round_num=2):
        # logger.debug(first)
        # logger.debug(last)
        if first is not None and last is not None \
            and (isinstance(first, str) or isinstance(first, float) or isinstance(first, int)) \
            and (isinstance(last, str) or isinstance(last, float) or isinstance(last, int)):
            first = float(first)
            last = float(last)
            if first == 0:
                first = 0.0000001
            # logger.debug("calc_perc() - done")
            return round((last-first) / (first/100), round_num)
        
        
        else:
            return None
        
       
        
        
 
    
    @staticmethod
    def calc_flpd(sub_data, column = "close"):
       if column == "close":
            return round((sub_data.iloc[-1].close - sub_data.iloc[0].open) / (sub_data.iloc[0].open/100), 2)
       else:
           return round((sub_data.iloc[-1][column] - sub_data.iloc[0][column]) / (sub_data.iloc[0][column]/100), 2)
           
    
    @staticmethod
    def zero_one_vals(min, max, val):
        return ((val - min) / (max - min))
    
    @staticmethod
    def add_first_last_perc_diff(data, column = "close"):
        if "flpd" in data:
            return data
        
        out = None
        if data is not None and len(data)>0:
            # symbols = data.groupby(by="sym")w
            flpd = (data.groupby(by="sym").last()[column]-data.groupby(
                by="sym").first()[column]) / (data.groupby(by="sym").first()[column]/100)
          
            out = pd.merge(data, flpd.round(2), on=[
                           "sym"], how="right", right_index=True)
            out.rename(columns={"close_y":"flpd","close_x":"close"},inplace=True)
            # logger.debug(out.iloc[0])
        
        return out
            
          
    @staticmethod
    def human_format(num):
        if not pd.isnull(num) and not isinstance(num, Timestamp) and not isinstance(num, datetime):
            num = float(num)
        else:
            return num
        
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        # add more suffixes if you need them
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
