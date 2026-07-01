import time 

class Timer():
    """Init signature: my_timer(mode="sec" or "hms")
            mode = "sec"(default) or "hms"
        after init like "timer = Timer()",
            use "timer.start()","timer.stop()" method to count the time,
                and use "time"
    """
    def __init__(self, mode = "sec"):
        assert isinstance(mode,str)
        self.mode  = mode
        self.time  = None
        self.start_time = None
        self.end_time   = None
    
    def __repr__(self):
        if self.time != None:
            if self.mode == "hms":
                return " running time = " + str(self.time[0])+ "小时" +\
                            str(self.time[1]) + "分钟" + str(self.time[2]) + "秒"
            return " running time = " + str(self.time)
        else:
            return " Warning................"

    __str__ = __repr__
    
    def __call__(self, description=''):
        print(description, ",", self.__repr__())

    def start(self):
        self.start_time = time.time()
        self.time  = None
    
    def stop(self):
        if self.start_time != None:
            self.end_time = time.time()
            self.__calc()
        else:
            print("请先开始计时")
            
    def __calc(self): 
        self.time  = self.end_time - self.start_time 
        if self.mode == "hms":
            second  = self.time%60
            minute = (self.time//60)%60
            hour = (self.time//3600)%60
            self.time = (hour,minute,second)
        self.end_time   = None
        self.start_time = None #清零
        
    