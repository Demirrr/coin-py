from apscheduler.schedulers.blocking import BlockingScheduler

sched = BlockingScheduler()

@sched.scheduled_job('interval', minutes=1)
def coin_news():
    # (1) Get new data
    # (2) Check existing data
    # (3) Compare how prices are changed in the last hour
    # (4) Write an email to the person
    print('This job is run every hour seconds.')


@sched.scheduled_job('interval', seconds=10)
def timed_job():
    print('This job is run every 10 seconds.')

@sched.scheduled_job('cron', day_of_week='mon-fri', hour=19)
def scheduled_job():
    print('This job is run every weekday at 10am.')

sched.configure()
sched.start()