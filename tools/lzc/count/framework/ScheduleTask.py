class ScheduleTask:
    # loops == -1为无限循环
    def __init__(self, interval: int, loops: int, callback: callable):
        self.interval = interval
        self.loops = loops
        self.callback = callback
        self.time_flag = 0
        self.now_loop = 0
        self.done = False

    def schedule_update(self, now):
        if not self.done:
            if self.time_flag == 0:
                self.time_flag = now
            if now - self.time_flag >= self.interval:
                if self.callback is not None:
                    self.callback()
                self.now_loop += 1
                if self.loops != -1 and self.now_loop >= self.loops:
                    self.done = True



