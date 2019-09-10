import time
import datetime
import logging


def log_make(name, log_folder, log_level):


	global logger

	
	dt_now = datetime.datetime.now()
	log_time= str(dt_now.year) + str(dt_now.month).zfill(2) + str(dt_now.day).zfill(2) 
	log_time2 =  str(dt_now.hour).zfill(2) + str(dt_now.minute).zfill(2) + str(dt_now.second).zfill(2)
	log_name = log_folder + log_time + log_time2

	logger = logging.getLogger(name)
	fmt = "%(asctime)s %(levelname)s %(funcName)s :%(message)s"
	logging.basicConfig(filename='%s.log' % log_name, level=log_level, format=fmt)

	console = logging.StreamHandler()
	# 個別にformatの形式を変えられる
	console_formatter = logging.Formatter(fmt)
	console.setFormatter(console_formatter)
	# sys.stderrにはざっくりとしたerror情報で良いので、INFOとする
	console.setLevel(log_level)
	# consoleという設定logging設定ができたので、適用したいmoduleに対して、addHandlerにその設定を突っ込む
	logger.addHandler(console)

	return logger





