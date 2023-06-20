import json


def report_time(start, end):
    print("-----------------------------------------------------------")
    print(
        "Start date time: {0}\nEnd date time: {1}\nTotal running time: {2}.".format(
            start, end, end - start
        )
    )


def get_anchors_from_json(path):
    with open(path, "r") as outfile:
        anchors_dict = json.load(outfile)

    return anchors_dict
