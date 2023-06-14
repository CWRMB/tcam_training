import argparse
import os
import random
import pickle
import time

parser = argparse.ArgumentParser()
parser.add_argument('--directory', default='/shared/data/Traffickcam')
parser.add_argument('--info', default='/shared/data/Traffickcam/hotelimageinfo.pkl')
args = parser.parse_args()

if __name__ == '__main__':
    with open(args.info, 'rb') as f:
        info = pickle.load(f)

    start = time.time()
    random.seed(0)
    hotel2img = {}
    i = 0
    for root, subdirs, filenames in sorted(os.walk(args.directory)):
        for filename in filenames:
           # image_id, hotel_id = str(filename.split('_')[0]), str(filename.split('_')[1].split('.')[0])
            filename_parts = filename.split('_')
            if len(filename_parts) < 2:
                print("Skipping file with unexepected format:", filename)
                continue
            image_id, hotel_id = str(filename_parts[0]), str(filename_parts[1].split('.')[0])
            try:
                capture_method = info[info['id'] == int(image_id)]['capture_method_id'].values[0]
            except ValueError:
                print("Error occured for image ID:", image_id)
                continue
            path = os.path.join(os.sep.join(root.split(os.sep)), filename)
            if os.path.getsize(path) > 0:
                if hotel_id not in hotel2img:
                    hotel2img[hotel_id] = [{'img': path, 'capture_method': capture_method}]
                else:
                    hotel2img[hotel_id].append({'img': path, 'capture_method': capture_method})
            print("File #{}".format(i))
            i += 1

    print("Loaded filepaths, time =", time.time() - start)

    hotels = list(hotel2img.keys())
    query_hotels = set()

    while len(query_hotels) < 15000:
        random_hotel = random.choice(hotels)
        tcam_imgs = [pair['img'] for pair in hotel2img[random_hotel] if pair['capture_method'] == 1]
        if len(tcam_imgs) >= 3:
            query_hotels.add(random_hotel)
            print("Added hotel #{} to queries".format(len(query_hotels)))

    print("Selected gallery / query hotels, time =", time.time() - start)

    query_hotels = list(query_hotels)
    random.shuffle(query_hotels)

    gallery, train_queries, validation_queries = [], [], []
    for hotel in query_hotels:
        tcam_imgs = [pair['img'] for pair in hotel2img[hotel] if pair['capture_method'] == 1]
        split_len = len(tcam_imgs) // 3
        random.shuffle(tcam_imgs)
        train_queries += tcam_imgs[:split_len]
        validation_queries += tcam_imgs[split_len: 2 * split_len]
        gallery += tcam_imgs[2 * split_len:]

    # start with just train query hotels
    train = gallery + train_queries

    print("Selected queries, time =", time.time() - start)

    for hotel in hotel2img:
        if hotel not in query_hotels:
            train += [pair['img'] for pair in hotel2img[hotel]]

    gallery = set(gallery)
    train_queries = set(train_queries)
    validation_queries = set(validation_queries)
    print("Finished splitting, time =", time.time() - start)
    print("gallery, train_queries, validation_queries")
    print(len(gallery), len(train_queries), len(validation_queries))

    d = {'gallery_imgs.dat': gallery, 'train_imgs.dat': train, 'validation_queries.dat': validation_queries,
         'train_queries.dat': train_queries}

    assert gallery.isdisjoint(train_queries) and \
           gallery.isdisjoint(validation_queries) and \
           validation_queries.isdisjoint(train_queries)

    for filename in d:
        with open(filename, 'wb') as f:
            pickle.dump(d[filename], f)
            f.close()
