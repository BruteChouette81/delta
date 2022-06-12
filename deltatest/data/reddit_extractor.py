import praw
# 1: 0WnreCeueQUh4amAfCDpvg

# 2: GPOr10YWXcF00IqeTEbQX1vI6-MZQA
# https://praw.readthedocs.io/en/stable/tutorials/comments.html



reddit = praw.Reddit(client_id="0WnreCeueQUh4amAfCDpvg", client_secret="GPOr10YWXcF00IqeTEbQX1vI6-MZQA", user_agent="conversation extractor")



### test
# we will get a list of subreddit
# for a number of post in this subreddit, extract all top comments
# create a dataset of structure {subreddit:
#                                   post:
#                                      conv-id{
#                                           1:top_comment,
#                                           2: replie
#                                           3: replie... 
#                                           }
#                                    } 
#  #

post = reddit.submission(url = "https://www.reddit.com/r/teslamotors/comments/u059s0/austin_factory_layout_was_flipped_180_on_site/")


text = {"conv" : [

]

}

id_list = []

count = 0
post.comments.replace_more(limit=None)
for comment in post.comments.list():
    if text["conv"]:
        id_list = []
        for conversation in text["conv"]:
            id_list.append(conversation["id"])

        print(len(id_list))
        if str(comment.parent_id).replace("t1_", "") not in id_list:
            conv = {
                "text": comment.body,
                "id": str(comment.id).replace("t1_", "")
            }

            text["conv"] += [conv]

        else:
            count_conv = 0
            for conversation in text["conv"]:
                if conversation["id"] == str(comment.parent_id).replace("t1_", ""):
                    if len(conversation) == 2:
                        text["conv"][count_conv]["response"] = str(comment.body)
                    break
                count_conv+=1
                        
    else:
        conv = {
                "text": comment.body,
                "id": str(comment.id).replace("t1_", "")
            }

        text["conv"] += [conv]
        print(text)

    count+=1

print(count)
print(text["conv"])
