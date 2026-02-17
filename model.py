from icrawler.builtin import BingImageCrawler
crawler = BingImageCrawler(storage={'root_dir': 'dataset/stairs'})

crawler.crawl(
    keyword='stairs images close outdoor ',
    max_num=100
)
