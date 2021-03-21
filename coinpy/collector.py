import cbpro
import pandas as pd
from .util import create_experiment_folder


class Collector:
    def __init__(self):
        self.__public_client = cbpro.PublicClient()

    def fetch(self, id_prod, granularity=300):
        """
        Timestamp	Epoch timestamp in milliseconds. You can learn more about timestamps, including how to convert them to human readable form, here.
        Open:	Opening price of the time interval in quote currency

        High:	Highest price reached during time interval, in quote currency.

        Low:    Lowest price reached during time interval, in quote currency.

        Close:	Closing price of the time interval, in the quote currency.

        Volume	Quantity of asset bought or sold, displayed in base currency.
        :param granularity:
        :param id_prod:
        :return:
        """
        # 300 seconds , 5 minutes, 300 data points => 1500 minutes => 25 hours.
        data = self.__public_client.get_product_historic_rates(id_prod, granularity=granularity)

        cols = [feat + '_' + id_prod if feat != 'time' else feat for feat in
                ['time', 'low', 'high', 'open', 'close', 'volume']]
        df = pd.DataFrame(data, columns=cols)
        # Unix-time to
        df.index = pd.to_datetime(df.time, unit='s')
        df.drop(columns=['time'], inplace=True)
        df = df.sort_values(by=['time'])

        return df

    def fetch_and_save(self, products=None, path=None, granularity=300):
        """

        :param products:
        :param path:
        :param granularity: acceptedGrans = [60, 300, 900, 3600, 21600, 86400]
        :return:
        """
        full_path = create_experiment_folder(path)
        print(f'Fetching data and will be saved in {full_path}')

        if products is None:
            # get all products
            for i in self.__public_client.get_products():
                id_prod = i['id']
                # @TODO Multiprocessing
                df = self.fetch(id_prod=id_prod,granularity=granularity)
                df.to_csv(f'{full_path}/{id_prod}.csv')
                print(f'{id_prod} saved.')
