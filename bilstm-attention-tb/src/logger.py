# -*- coding: utf-8 -*
import logging
#设置日志的级别和格式
logging.basicConfig(level=logging.DEBUG,filename='barchybrid/log/log.txt',format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)