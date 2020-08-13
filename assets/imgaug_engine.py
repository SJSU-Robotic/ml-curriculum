#!/usr/bin/env python
# auth ntwong0

import os, errno, math, argparse
from time import perf_counter # used to log performance/timing for generate_images()
from PIL import Image, ImageOps
import numpy as np 

""" 
# _TODO_ apply imgaug augmentations
import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.random as iar

ia.seed(123)

def prismatize_imgaug(images, random_state, parents, hooks):
    ret_images = list()
    for image in images:
        val    = random_state.uniform(0.0001, 0.9999)
        print(val)
        aug1   = iaa.Resize({"width": val})
        image1 = aug1(images=image.reshape(1,*image.shape)).reshape(image.shape[0],-1,1)
        aug2   = iaa.Resize({"width": 1 - val})
        image2 = aug2(images=image.reshape(1,*image.shape)).reshape(image.shape[0],-1,1)
        ret_images.append(np.concatenate((image1, image2), axis=1).reshape(image.shape[0],-1,1))
    # ret_images =  np.stack(ret_images, axis=0).reshape(len(ret_images), *ret_images[0].shape)
    # print(ret_images.shape)
    # return ret_images
    return np.stack(ret_images, axis=0).reshape(len(ret_images), *ret_images[0].shape)

def ndarray_from_filename(dirname, filename):
    return np.asarray(Image.open(dirname + "/" + filename))

# if not_ar_tags=False, then image is greyscale and height/width does not vary across images
# def load_from_dir(dirname, not_ar_tags=True):
def load_from_dir(dirname):
    return [pillow_from_filename(dirname, fn) for fn in os.listdir(dirname)]
    # mats = [ndarray_from_filename(dirname, fn) for fn in os.listdir(dirname)]
    # if not_ar_tags:
    #     return mats
    # mat = np.stack(mats, axis=0)
    # return mat.reshape(*mat.shape, 4 if not_ar_tags else 1)

aug = iaa.Lambda(prismatize_imgaug)

# sanity check
# for mat in [mat.reshape(mat.shape[0], mat.shape[1]) for mat in aug(images=load_from_dir("ar_tags"))]:
#     Image.fromarray(mat).show()

# note:
# pillow.Image.size: (width, height, channels)
# np.ndarray.shape:  (height, width, channels)
"""

""" unoptimized (don't use)

# get tags and resize
ar_tags = load_from_dir(get_dat_dir(dir_ar_tags, dir_images))
for ar_tag in ar_tags: 
    ar_tag.thumbnail((ar_tag_width_max, ar_tag_width_max), Image.ANTIALIAS)

# get posts and resize
posts = load_from_dir(get_dat_dir(dir_posts, dir_images))
for post in posts: 
    post.thumbnail((post_height_max, post_height_max), Image.ANTIALIAS)

# pad posts to match tag width
posts = [ImageOps.expand(post, 
            (
                (ar_tag_width_max - post.size[0])//2, 
                0, 
                (ar_tag_width_max - post.size[0])//2 + (1 if post.size[0] % 2 else 0), 
                0
            )
         ) for post in posts]
"""
### 

# _DONE_ change the way files are processed through pillow before np.stacked for imgaug
# bottleneck: file open limit, so instead
# load file, modify, convert to ndarray, then np.stack
# example: processing for humans

def prismatize(
    tag_np, 
    scale_factor, 
    ret_type="pil" # "np" to return numpy.ndarray, or "pil" to return PIL.PngImagePlugin.PngImageFile
):
    try:
        image_left   = Image.fromarray(tag_np)
        image_left   = image_left.resize((int(image_left.size[0] * scale_factor), image_left.size[1]), Image.ANTIALIAS)
        image_right  = Image.fromarray(tag_np)
        image_right  = image_right.resize((int(image_right.size[0] * (1 - scale_factor)), image_right.size[1]), Image.ANTIALIAS)
        padding      = (image_left.size[0], 0, 0, 0)
        image_right  = ImageOps.expand(image_right, padding)
        image        = Image.composite(image_left, image_right, image_left)
        return np.asarray(image) if ret_type is "np" else image
    except ValueError: # scale_factor too low / too high to create a second image for prismatization
        return tag_np if ret_type is "np" else Image.fromarray(tag_np)

def crop_and_pad_post(
    post_np,
    scale_factor,
    tag_height
):
    post_img = Image.fromarray(post_np)
    # randomize post height
    scale_factor  = np.random.uniform(0.3, 1)
    if scale_factor != 1:
        post_img = post_img.crop((0, post_img.size[1] * (1 - scale_factor), post_img.size[0], post_img.size[1]))
    # pad the post to accommodate tag overlay
    return ImageOps.expand(post_img, (0, tag_height, 0, 0))

def get_dat_dir(dir_cat, dir_dat):
    return dir_cat + "/" + dir_dat

def pillow_from_filename(dirname, filename):
    return Image.open(dirname + "/" + filename)

# output shape is (height, width, 4), corresponding to RGBA format for .png's with alpha
def ndarray_from_pillow(image, is_grayscale=False):
    if is_grayscale:
        return np.asarray(image.convert("RGBA"))
    else:
        return np.asarray(image)

""" load_data
        Loads data from various directories into memory. Accommodations must be made to limit the number of open 
        filesystem hooks due to Image.open(), therefore we convert them to np.ndarrays
    outputs:
        tags 
        posts
        humans
        bgs
        bg_metas
"""
def load_data(
    # directory params
    dir_parent,
    dir_tags        = "ar_tags",
    dir_posts       = "posts",
    dir_humans      = "humans",
    dir_bgs         = "desert_backgrounds",
    dir_images      = "images",
    dir_annotations = "annotations",
    # pixel params
    _720p_height     = 720,
    _720p_width      = 1280,
    _1440p_height    = 1440,
    _1440p_width     = 2560,
    post_height_max  = 144,  # 10% of _1440p_height
    tag_width_max    = 28,   # 20% of post_height_max
    human_height_max = 288   # 20% of _1440p_height
):
    # chdir to parent directory
    os.chdir(dir_parent)
    # get tags and resize
    tags = list()
    tags_fns = sorted(os.listdir(get_dat_dir(dir_tags, dir_images)))
    for tags_fn in tags_fns:
        if "._" in tags_fn: continue # ignore macOS metadata files
        tag = pillow_from_filename(get_dat_dir(dir_tags, dir_images), tags_fn)
        tag.thumbnail((tag_width_max, tag_width_max), Image.ANTIALIAS)
        tags.append(ndarray_from_pillow(tag, is_grayscale=True))
    # get posts and resize
    posts = list()
    posts_fns = sorted(os.listdir(get_dat_dir(dir_posts, dir_images)))
    for posts_fn in posts_fns:
        if "._" in posts_fn: continue # ignore macOS metadata files
        post = pillow_from_filename(get_dat_dir(dir_posts, dir_images), posts_fn)
        post.thumbnail((post_height_max, post_height_max), Image.ANTIALIAS)
        # prepare post for tag overlay
        post = ImageOps.expand(post, 
                    (
                        (tag_width_max - post.size[0])//2, 
                        0, 
                        (tag_width_max - post.size[0])//2 + (1 if post.size[0] % 2 else 0), 
                        0
                    )
                )
        posts.append(ndarray_from_pillow(post))
    # get humans and resize
    humans = list()
    humans_fns = sorted(os.listdir(get_dat_dir(dir_humans, dir_images)))
    for human_fn in humans_fns:
        if "._" in human_fn: continue # ignore macOS metadata files
        human = pillow_from_filename(get_dat_dir(dir_humans, dir_images),human_fn)
        ratio = human_height_max/human.size[1]
        human = human.resize((int(human.size[0] * ratio), int(human.size[1] * ratio)))
        humans.append(ndarray_from_pillow(human))
    # get backgrounds and resize
    bgs, bgs_ratios = list(), list()
    bgs_fns = sorted(os.listdir(get_dat_dir(dir_bgs, dir_images)))
    for bgs_fn in bgs_fns:
        if "._" in bgs_fn: continue # ignore macOS metadata files
        bg           = pillow_from_filename(get_dat_dir(dir_bgs, dir_images), bgs_fn)
        ratio_width  = 1 if bg.size[0] >= _1440p_width else _1440p_width / bg.size[0] # scale up based on width
        ratio_height = 1 if bg.size[1] >= _1440p_height else _1440p_height / bg.size[1] # scale up based on height
        ratio        = ratio_width if ratio_width >= ratio_height else ratio_height
        bgs_ratios.append(ratio)
        bg           = bg.resize((int(bg.size[0] * ratio), int(bg.size[1] * ratio)))
        bgs.append(ndarray_from_pillow(bg))
    # get background annotations, extract distances to pixels mapping
    # header: bottom_right_x, bottom_right_y, top_left_x, top_left_y, dist_bottom, dist_top
    bg_metas = list()
    bg_meta_fns = sorted(os.listdir(get_dat_dir(dir_bgs, dir_annotations)))
    index = 0 # use this to access the corresponding bgs_ratios element
    for bg_meta_fn in bg_meta_fns:
        if "._" in bg_meta_fn: continue # ignore macOS metadata files
        bg_meta = np.genfromtxt(get_dat_dir(dir_bgs, dir_annotations)+"/"+bg_meta_fn, 
                                    delimiter=',',
                                    skip_header=1)
        # update annotations via scaling ratios from above
        ratio = bgs_ratios[index] 
        ratio = np.array([ratio, ratio, ratio, ratio, 1, 1])
        bg_meta = np.floor(bg_meta * ratio)
        index += 1
        # append onto bg_metas
        bg_metas.append(bg_meta)
    # return data
    return tags, posts, humans, bgs, bg_metas

# load_data(os.getcwd(), dir_tags, dir_posts, dir_humans, dir_bgs, dir_images, dir_annotations)
# tags, posts, humans, bgs, bg_metas = load_data(os.getcwd(), dir_tags, dir_posts, dir_humans, dir_bgs, dir_images, dir_annotations)

# image_nps, annotations = generate_images(tags, posts, humans, bgs, bg_metas)
""" generate_images
    outputs: 
        if storage_setting == "filesystem":
            None
        else:
            image_nps:   list of all np.ndarray'd images generated, len(images) == count
            annotations: list of all image meta generated, len(annotations) == count
"""
def generate_images(
    # inputs
    tags,
    posts,
    humans,
    bgs,
    bg_metas,
    # params
    gen_type        = "full",        # full  : 720x1280 
    #                                # square: 28x28
    count           = 30,            # images to generate
    seed            = 0,
    verbosity       = 1,             # 0: no verbosity, 
    #                                # 1: print filename and annotations to console
    #                                # 2: previous + show images (avoid for count vars > 30)
    storage_setting = "memory",      # where will generated images and annotations be stored?
    #                                #  filesystem: store to filesystem at dir_dest
    #                                #  memory:     store and output 
    #                                #  both:       
    # dir params
    dir_images      = "images",
    dir_annotations = "annotations",
    dir_dest        = "generated",
    # pixel parameters
    square_width     = 28,   # used if gen_type == "square"
                             # if square_width <= 0: skip resize
    _720p_height     = 720,
    _720p_width      = 1280,
    _1440p_height    = 1440,
    _1440p_width     = 2560,
    post_height_max  = 144,  # 10% of _1440p_height
    tag_width_max    = 28,   # 20% of post_height_max
    human_height_max = 288   # 20% of _1440p_height
):
    # make dir_dest if not exist
    try:
        os.makedirs(dir_dest)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    # make dir_images if not exist
    try:
        os.makedirs("%s/%s" % (dir_dest, dir_images))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    # make dir_annotations if not exist
    try:
        os.makedirs("%s/%s" % (dir_dest, dir_annotations))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    # init randomizer
    np.random.seed(seed)
    # initialize buckets based on storage_setting
    image_nps   = list() if storage_setting != "filesystem" else None
    annotations = list() if storage_setting != "filesystem" else None
    # generate by looping
    # _TODO_ parallelize this
    t_prev = perf_counter()
    for idx in range(count):
        # select a background
        bg_idx = np.random.randint(0, len(bgs))
        bg, bg_meta = bgs[bg_idx], bg_metas[bg_idx]
        # select a background sector
        sector_idx = np.random.randint(0, len(bg_meta))
        sector_np = bg_meta[sector_idx]
        sector = dict()
        sector["x_right"], \
        sector["y_bottom"], \
        sector["x_left"], \
        sector["y_top"], \
        sector["dist_bottom"], \
        sector["dist_top"] = np.split(sector_np, sector_np.shape[0])
        # select a background crop, inclusive of sector
        #   select a pixel within sector to center the crop on
        #   break if error in annotations
        #   _TODO_ should happen much earlier, consider moving to load_data(), unless deprecated via reciprocal drop-off approach
        try:
            crop_x = int(np.random.randint(sector["x_left"], sector["x_right"]))
        except ValueError:
            print("ValueError with bg_idx: %d, x_left: %d, x_right: %d" % (bg_idx, sector["x_left"], sector["x_right"]))
            break
        try:
            crop_y = int(np.random.randint(sector["y_top"],  sector["y_bottom"]))
        except ValueError:
            print("ValueError with bg_idx: %d, y_top: %d, y_bottom: %d" % (bg_idx, sector["y_top"], sector["y_bottom"]))
            break
        #   try to center crop around the current pixel
        crop = dict()
        #   determine crop_x boundaries
        if (crop_x - _720p_width // 2) < 0:
            crop["x_left"]  = 0
            crop["x_right"] = _720p_width - 1
        elif (crop_x + _720p_width // 2) > bg.shape[1]:
            crop["x_left"]  = bg.shape[1] - _720p_width
            crop["x_right"] = bg.shape[1] - 1
        else:
            crop["x_left"]  = crop_x - _720p_width // 2
            crop["x_right"] = crop_x + _720p_width // 2 - 1
        #   determine crop_y boundaries
        if (crop_y - _720p_height // 2) < 0:
            crop["y_top"]    = 0
            crop["y_bottom"] = _720p_height - 1
        elif (crop_y + _720p_height // 2) > bg.shape[1]:
            crop["y_top"]    = bg.shape[0] - _720p_height
            crop["y_bottom"] = bg.shape[0] - 1
        else:
            crop["y_top"]    = crop_y - _720p_height // 2
            crop["y_bottom"] = crop_y + _720p_height // 2 - 1
        # select a insertion point
        #   randomly select a pixel at which to insert the image
        #   this pixel corresponds to the bottom left pixel of the image to be inserted
        #   this differs from convention (using top left pixel) because y_bottom is used to compute distance
        #   determine x position, y position, which must lie within crop and within sector
        insertion_x  = np.random.randint(
            crop["x_left"]  if sector["x_left"] < crop["x_left"]    else sector["x_left"], 
            crop["x_right"] if crop["x_right"]  < sector["x_right"] else sector["x_right"]
        )
        insertion_y  = np.random.randint(
            crop["y_top"]    if sector["y_top"]  < crop["y_top"]      else sector["y_top"],
            crop["y_bottom"] if crop["y_bottom"] < sector["y_bottom"] else sector["y_bottom"]
        )
        # determine what to insert, then pillowize into image
        insertion_img = None
        selection = ("human", "post", "gate")[np.random.randint(3)]
        if selection is "human":
            insertion_img = Image.fromarray(humans[np.random.randint(len(humans))])
        elif selection is "post":
            tag_np   = tags[np.random.randint(len(tags))]
            tag_img  = prismatize(tag_np, np.random.uniform(0, 1))
            post_np  = posts[np.random.randint(len(posts))]
            post_img = crop_and_pad_post(post_np, np.random.uniform(0.3, 1), tag_img.size[1])
            insertion_img = Image.composite(tag_img, post_img, tag_img)
        else: # selection is "gate"
            tag0_idx = np.random.randint(3, 11)
            tag0_img = prismatize(tags[tag0_idx], np.random.uniform(0, 1))
            tag1_idx = tag0_idx + 1 if tag0_idx % 2 else tag0_idx - 1
            tag1_img = prismatize(tags[tag1_idx], np.random.uniform(0, 1))
            post_np  = posts[np.random.randint(len(posts))]
            post_img = crop_and_pad_post(post_np, np.random.uniform(0.3, 1), tag_img.size[1])
            tag0_img = Image.composite(tag0_img, post_img, tag0_img)
            tag1_img = Image.composite(tag1_img, post_img, tag1_img)
            # pad tag0+post, then overlay tag1 onto tag0+post
            gate_width    = np.random.randint(post_height_max * 2 + post_img.size[0])
            tag0_img      = ImageOps.expand(tag0_img, (gate_width, 0, 0, 0))
            insertion_img = Image.composite(tag1_img, tag0_img, tag1_img)
        # scale the image
        dist  = ((sector["dist_top"] - sector["dist_bottom"]) \
                / (sector["y_top"] - sector["y_bottom"])) \
                * (insertion_y - sector["y_bottom"]) + sector["dist_bottom"]
        scale = (1/(2**(sector_idx+1)) - 1/(2**(sector_idx))) \
                / (sector["dist_top"] - sector["dist_bottom"]) \
                * (dist - sector["dist_bottom"]) + 1/(2**(sector_idx))
        # resize image based on y position and distance
        insertion_img = insertion_img.resize((int(insertion_img.size[0] * scale), int(insertion_img.size[1] * scale)))
        # establish safety bounds
        safety = {
            "x_left"  : crop["x_left"],
            "y_top"   : crop["y_top"]   - insertion_img.size[0],
            "x_right" : crop["x_right"] - insertion_img.size[1], 
            "y_bottom": crop["y_bottom"]
        }
        #   if insertion point out of safety bounds, set them onto safety bounds
        insertion_x = int(insertion_x if insertion_x < safety["x_right"] else safety["x_right"])
        insertion_y = int(insertion_y if insertion_y > safety["y_top"]   else safety["y_top"])
        # insert the image
        #   place image onto desert_bg
        bg_img = Image.fromarray(bg)
        padding = (insertion_x, insertion_y - insertion_img.size[1], 0, 0)
        insertion_img_padded = ImageOps.expand(insertion_img, padding)
        overlay = Image.composite(insertion_img_padded, bg_img, insertion_img_padded)
        # apply cropping onto overlay
        if gen_type == "full":
            # update crop region within bounds of bg_img
            #   adjust horizontal crop
            if crop["x_left"] < 0: # not likely
                crop["x_right"] += 0 - crop["x_left"]
                crop["x_left"] = 0
            elif crop["x_right"] > bg_img.size[0]:
                crop["x_left"] -= crop["x_right"] - bg_img.size[0]
                crop["x_right"] = bg_img.size[0]
            else:
                pass # nothing wrong with horizontal crop
            #   adjust vertical crop
            if crop["y_top"] < 0: # not likely
                crop["y_bottom"] += 0 - crop["y_top"]
                crop["y_top"] = 0
            elif crop["y_bottom"] > bg_img.size[1]:
                crop["y_top"] -= crop["y_bottom"] - bg_img.size[1]
                crop["y_bottom"] = bg_img.size[1]
            # update insertion point based on crop region
            insertion_x, insertion_y = insertion_x - crop["x_left"], insertion_y - crop["y_top"]
            overlay = overlay.crop((crop["x_left"], crop["y_top"], crop["x_right"], crop["y_bottom"]))
        else: # gen_type == "square"
            # grab square crop centered on insertion_img
            #   insertion_img is wider than it is tall, possible for gates
            if insertion_img.size[0] >= insertion_img.size[1]:
                # use the width
                width = insertion_img.size[0]
                # if we fall below the limit of crop["y_top"], just use crop["y_top"] instead
                top = crop["y_top"] if (insertion_y - width < crop["y_top"]) else (insertion_y - width)
                crop["x_left"], crop["y_top"], crop["x_right"], crop["y_bottom"] = insertion_x, top, insertion_x + width, top + width
            else: # insertion_img is taller than it is wide
                # use the height
                height = insertion_img.size[1]
                # find the leftmost bounds for our crop, which would otherwise be x_insertion
                leftmost_bound = insertion_x + insertion_img.size[0] - height
                leftmost_bound = leftmost_bound if crop["x_left"] < leftmost_bound else crop["x_left"]
                try:
                    left = leftmost_bound if leftmost_bound == insertion_x else np.random.randint(leftmost_bound, insertion_x)
                except ValueError:
                    print(
                        "ValueError with bg_idx: %d, leftmost_bound: %d, height: %d, insertion_x: %d, insertion_img.size: (%d, %d), crop[\"x_left\"]: %d" % \
                        (bg_idx, leftmost_bound, height, insertion_x, insertion_img.size[0], insertion_img.size[1], crop["x_left"])
                    )
                    break
                crop["x_left"], crop["y_top"], crop["x_right"], crop["y_bottom"] = left, insertion_y - height, left + height, insertion_y
            # update crop region within bounds of bg_img
            #   adjust horizontal crop
            if crop["x_left"] < 0: # not likely
                crop["x_right"] += 0 - crop["x_left"]
                crop["x_left"] = 0
            elif crop["x_right"] > bg_img.size[0]:
                crop["x_left"] -= crop["x_right"] - bg_img.size[0]
                crop["x_right"] = bg_img.size[0]
            else:
                pass # nothing wrong with horizontal crop
            #   adjust vertical crop
            if crop["y_top"] < 0: # not likely
                crop["y_bottom"] += 0 - crop["y_top"]
                crop["y_top"] = 0
            elif crop["y_bottom"] > bg_img.size[1]:
                crop["y_top"] -= crop["y_bottom"] - bg_img.size[1]
                crop["y_bottom"] = bg_img.size[1]
            overlay = overlay.crop((crop["x_left"], crop["y_top"], crop["x_right"], crop["y_bottom"]))
            # resize based on width limitations, so long as square_width >= 1
            if square_width >= 1:
                overlay = overlay.resize((square_width, square_width))
        # prepare annotations
        fn_image, fn_meta = None, None
        meta = "%s, %d, %d, %d" % (selection, insertion_x, insertion_y, sector_idx) if gen_type == "full" else \
               "%s, %d" % (selection, sector_idx)
        # store images and annotations
        if storage_setting == "filesystem" or storage_setting == "both":
            # save image to dir_dest/dir_image
            fn_image = "%s/%s/%s%s.png" % (dir_dest, dir_images, dir_dest, str(idx).zfill(math.floor(math.log(count+1,10))+1))
            overlay.save(fn_image, "PNG")
            # save annotations to dir_dest/dir_annotations
            fn_meta = "%s/%s/%s%s.txt" % (dir_dest, dir_annotations, dir_dest, str(idx).zfill(math.floor(math.log(count+1,10))+1))
            f = open(fn_meta, "w")
            f.write(meta)
            f.close()
        if storage_setting == "memory" or storage_setting == "both":
            image_nps.append(np.asarray(overlay))
            annotations.append(meta)
        # handle logging verbosity
        if verbosity >= 1:
            print("index: %d, meta: " % idx + meta if fn_image is None else "filename: " + fn_image + ", meta: " + meta)
            print("bg_img.size: ", bg_img.size)
            print("crop: ", crop)
            # how long did this iteration take?
            t_curr = perf_counter()
            print("elapsed time: ", t_curr - t_prev, " s")
            t_prev = t_curr
            print('')
        if verbosity >= 2:
            overlay.show()
    # return data depending on storage_setting
    if storage_setting == "memory" or storage_setting == "both":
        return image_nps, annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gen_type", 
        help="generator type. full: 720p image. square: 28x28 image.", 
        choices=["full", "square"]
    )
    parser.add_argument(
        "count", 
        help="number of images to generate.", 
        type=int
    )
    parser.add_argument(
        "--seed", 
        default=0,
        help="seed value to initialize np.random.", 
        type=int
    )
    parser.add_argument(
        "--verbosity", 
        default=1,
        help="verbosity for logging image generator telemetry to console.", 
        choices=[0, 1, 2],
        type=int
    )
    parser.add_argument(
        "--storage_setting", 
        default="filesystem",
        help="determine where generated images and annotations are stored.", 
        choices=["filesystem", "memory", "both"]
    )
    # directory parameters
    parser.add_argument(
        "--dir_dataset", 
        default=os.getcwd(),
        help="directory of the dataset. use $PWD by default."
    )
    parser.add_argument(
        "--dir_tags",
        default="ar_tags",
        help="directory of the ALVAR AR tags."
    )
    parser.add_argument(
        "--dir_posts",
        default="posts",
        help="directory of posts."
    )
    parser.add_argument(
        "--dir_humans",
        default="humans",
        help="directory of pedestrians."
    )
    parser.add_argument(
        "--dir_bgs",
        default="desert_backgrounds",
        help="directory of desert backgrounds."
    )
    parser.add_argument(
        "--dir_images",
        default="images",
        help="directory of images."
    )
    parser.add_argument(
        "--dir_annotations",
        default="annotations",
        help="directory of image metadata."
    )
    parser.add_argument(
        "--dir_dest",
        default="generated",
        help="directory to which generated images are saved"
    )
    # pixel params
    parser.add_argument(
        "--square_width",
        default=28,
        help="relevant if gen_type == \"square\". skip resize if square_width <= 0",
        type=int
    )
    parser.add_argument(
        "--_720p_height",
        default=720,
        help="720p pixel height.",
        type=int
    )
    parser.add_argument(
        "--_720p_width",
        default=1280,
        help="720p pixel width.",
        type=int
    )
    parser.add_argument(
        "--_1440p_height",
        default=1440,
        help="1440p pixel height.",
        type=int
    )
    parser.add_argument(
        "--_1440p_width",
        default=2560,
        help="1440p pixel width.",
        type=int
    )
    parser.add_argument(
        "--post_height_max",
        default=144,  # 10% of _1440p_height
        help="max pixel height for post images.",
        type=int
    )
    parser.add_argument(
        "--tag_width_max",
        default=28,   # 20% of post_height_max
        help="max pixel height for tag images.",
        type=int
    )
    parser.add_argument(
        "--human_height_max",
        default=288,  # 20% of _1440p_height
        help="max pixel height for pedestrian images.",
        type=int
    )
    # load and validate args
    args = parser.parse_args()
    assert args.count            >  0
    assert args.seed             >= 0
    # assert args.square_width # already validated by parser
    assert args._720p_height     >  0
    assert args._720p_width      >  0
    assert args._1440p_height    >  0
    assert args._1440p_width     >  0
    assert args.post_height_max  >  0
    assert args.tag_width_max    >= 0
    assert args.human_height_max >  0
    
    if os.getcwd() != args.dir_dataset:
        os.chdir(args.dir_dataset)

    tags, posts, humans, bgs, bg_metas = load_data(
        # directories for data
        args.dir_dataset,
        args.dir_tags,
        args.dir_posts,
        args.dir_humans,
        args.dir_bgs,
        args.dir_images,
        args.dir_annotations,
        # parameters
        args._720p_height,
        args._720p_width,
        args._1440p_height,
        args._1440p_width,
        args.post_height_max,
        args.tag_width_max,
        args.human_height_max 
    )
    # generate_images(tags, posts, humans, bgs, bg_metas, gen_type="square", count=150)
    # image_nps, annotations = generate_images(
    ret_val = generate_images(
        # inputs
        tags,
        posts,
        humans,
        bgs,
        bg_metas,
        # params
        gen_type         = args.gen_type,
        count            = args.count,
        seed             = args.seed,
        verbosity        = args.verbosity,
        storage_setting  = args.storage_setting,
        # dir params
        dir_images       = args.dir_images,
        dir_annotations  = args.dir_annotations,
        dir_dest         = args.dir_dest,
        # more parameters
        square_width     = args.square_width,
        _720p_height     = args._720p_height,
        _720p_width      = args._720p_width,
        _1440p_height    = args._1440p_height,
        _1440p_width     = args._1440p_width,
        post_height_max  = args.post_height_max,
        tag_width_max    = args.tag_width_max,
        human_height_max = args.human_height_max
    )

    if args.storage_setting == "memory" or args.storage_setting == "both":
        image_nps, annotations = ret_val
        print("sampling from outputs of generate_image()")
        Image.fromarray(image_nps[0]).show()
        print(annotations[0])
