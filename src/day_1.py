from advent_of_code_2022 import INPUT_PATH
from colorama import Fore


def get_list_of_elves_calories() -> list:
    """"""
    input_file = INPUT_PATH / 'day_1_input_1.txt'
    with open(input_file, 'r') as f:
        raw_data = f.read()

    calories_lol = [
        [int(n) for n in elf_inv.split('\n')] for elf_inv in raw_data.split('\n\n')
    ]
    return calories_lol


def find_max_of_calories(calories_lol):
    return max(sum(elf_calories) for elf_calories in calories_lol)


def find_max_of_top_3_calories_providers(calories_lol):
    # sort lol of calories by sum
    sorted_calories_lol = sorted(calories_lol, key=lambda x: sum(x), reverse=True)

    return sum(sum(elem) for elem in sorted_calories_lol[:3])


def report_result(res: any, msg='result: '):
    print(msg.format(str(res)))


def solve_puzzle_1():
    calories_lol = get_list_of_elves_calories()
    max_cal = find_max_of_calories(calories_lol)
    msg = f"The elf with the most calories has a total of " + Fore.GREEN + "{}" + Fore.RESET + " calories."
    report_result(max_cal, msg)


def solve_puzzle_2():
    calories_lol = get_list_of_elves_calories()
    max_cal = find_max_of_top_3_calories_providers(calories_lol)
    msg = "The top 3 elves with the most calories have a total of " + Fore.GREEN + "{}" + Fore.RESET + " calories."
    report_result(max_cal, msg)


if __name__ == '__main__':
    solve_puzzle_1()
    solve_puzzle_2()
