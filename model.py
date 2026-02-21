from icrawler.builtin import BingImageCrawler
crawler = BingImageCrawler(storage={'root_dir': 'dataset/mouse'})

crawler.crawl(
    keyword='mouse rgb image computer',
    max_num=100
)
