import csv

new_file_name = "AffairsRB_new.csv"

rb_headers = ["identifier", "constant", "rating", "age", "yearsmarried",
              "number_of_children", "religiousness", "education", "not_used_1",
              "occupation", "husband_occupation", "extramarital_time_per_year",
              "not_used_2", "not_used_3"]


def process_each_line_rb(line):
    array_string = line[0].split(" ")
    result = []
    result2 = []
    for word in array_string:
        if word != "":
            result.append(word)
    for word in result:
        if word.endswith("."):
            word = word + "0"
        result2.append(word)
    return result2


def create_affairs_rb_csv():
    with open('AffairsRB.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        with open(new_file_name, 'w') as new_file:
            csv_writer = csv.writer(new_file, lineterminator='\n')
            csv_writer.writerow(rb_headers)
            for line in csv_reader:
                csv_writer.writerow(process_each_line_rb(line))
