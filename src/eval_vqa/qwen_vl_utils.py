def process_vision_info(messages):
    """
    Extract image and video inputs from a list of chat messages.
    Returns (images, videos) lists to feed into the processor.
    """
    images, videos = [], []
    for message in messages:
        for content in message[0]["content"]:
            if content["type"] == "image":
                images.append(content["image"])
            elif content["type"] == "video":
                videos.append(content["video"])
    return images, videos
