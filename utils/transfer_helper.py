import geohash2
from datetime import datetime, timedelta

def encodeLoc(loc):
    return geohash2.encode(loc[0], loc[1]) # loc[0]: latitude, loc[1]: longtitude 

def parseTime(time, time_format):
    '''
    parse to datetime
    '''
    # only implement 111111
    if time_format[0] == '111111':
        return datetime.strptime(time[0], '%Y-%m-%d-%H-%M-%S') # TODO: check if this is UTC Time ?

def calculateBaseTime(start_time, base_zero):
    if base_zero:
        return start_time - timedelta(hours=start_time.hour, minutes=start_time.minute, seconds=start_time.second,microseconds=start_time.microsecond)
    else:
        # time length = 12
        if start_time.hour < 12:
            return start_time - timedelta(hours=start_time.hour, minutes=start_time.minute, seconds=start_time.second,microseconds=start_time.microsecond)
        else:
            return start_time - timedelta(hours=start_time.hour - 12, minutes=start_time.minute, seconds=start_time.second,microseconds=start_time.microsecond)

def calculateTimeOff(now_time, base_time):
    # 先将 now 按小时对齐
    now_time = now_time - timedelta(minutes=now_time.minute, seconds=now_time.second)
    delta = now_time - base_time
    return delta.days * 24 + delta.seconds / 3600