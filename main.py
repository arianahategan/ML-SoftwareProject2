from knn import run_knn


def print_menu():
    menu_string = "\nAvailable commands:\n"
    menu_string += "\t 1. Get performance measures using KNNs (all attributes)\n"
    menu_string += "\t 2. Get performance measures using KNNs (reduced number of attributes)\n"

    menu_string += "\t 0. Exit\n"
    print(menu_string)


def run_console():
    while True:
        print_menu()
        command = int(input("Enter a command: "))
        try:
            if command == 1:
                number_neighbors = int(input("Enter the number of neighbors: "))
                run_knn(number_neighbors, 0)
            if command == 2:
                number_neighbors = int(input("Enter the number of neighbors: "))
                run_knn(number_neighbors, 1)
            elif command == 0:
                return
        except Exception as e:
            print(e)


run_console()
