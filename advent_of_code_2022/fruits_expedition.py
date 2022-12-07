import copy
import datetime
import string
from pathlib import Path
from colorama import Fore
from string import Formatter
import re
from itertools import zip_longest


from advent_of_code_2022.folder_tree import Folder, File
# from collections import Counter
from datetime import timedelta
# from typing import Union
# import re
# import pandas as pd
# import numpy as np
# from copy import deepcopy
# from typing import List
# import bisect
# import time
# import queue


class ExpeditionLeader:

    def __init__(self):
        """"""
        self.inputs_dir = Path(__file__).parent / 'inputs'
        print('leader spawned, ready to explore...')

    def do_some_exploring(self):
        self.report_start()

        # self.calories_lol = self.get_input_lol_of_calories()
        # self.max_individual_calories = self.find_max_of_calories(self.calories_lol)
        # self.report_result(
        #     self.max_individual_calories,
        #     prefix="The elf with the most calories has a total of",
        #     suffix="calories."
        # )
        # self.max_calories_for_3_top_elves = self.find_max_of_top_3_calories_providers(self.calories_lol)
        # self.report_result(
        #     self.max_calories_for_3_top_elves,
        #     prefix="The top 3 elves with the most calories have a total of",
        #     suffix=" calories."
        #
        # )
        #
        # self.rucksacks_inventories = self.get_input_rucksacks_iventory()
        # self.priority_sum = self.find_priority_sum_of_misplaced_items(self.rucksacks_inventories)
        # self.report_result(
        #     self.priority_sum,
        #     prefix="The sum of the priority of items misplaced in rucksacks is"
        # )
        # self.priority_sum_badges = self.find_priority_sum_of_badges_items(self.rucksacks_inventories)
        # self.report_result(
        #     self.priority_sum_badges,
        #     prefix="The sum of the priority of badges items in rucksacks is"
        # )
        #
        # self.pair_section_ranges = self.get_input_cleaning_duty_sections_assignments()
        # self.n_redundant_assignments = self.find_cleaning_duty_redundant_assignments(self.pair_section_ranges)
        # self.report_result(
        #     self.n_redundant_assignments,
        #     prefix="The number of redundant assignment pairs is"
        # )
        #
        # self.n_overlapping_assigments = self.find_cleaning_duty_overlapping_assignments(self.pair_section_ranges)
        # self.report_result(
        #     self.n_overlapping_assigments,
        #     prefix="The number of overlapping assignment pairs is"
        # )
        #
        # self.d5_stacks, self.d5_program = self.get_input_crates_rearrangement_planning()
        # self.top_crates_9000 = self.find_top_crates_after_rearrangement(self.d5_stacks, self.d5_program)
        # self.report_result(
        #     self.top_crates_9000,
        #     prefix="The top crates after rearrangement with crane model 9000 are"
        # )
        # self.top_crates_9001 = self.find_top_crates_after_rearrangement(self.d5_stacks, self.d5_program, crane_model=9001)
        # self.report_result(
        #     self.top_crates_9001,
        #     prefix="The top crates after rearrangement with crane model 9001 are"
        # )
        #
        # self.d6_datastream = self.get_input_d6_datasream_buffer()
        # self.n_char_before_sop = self.count_chars_before_marker(self.d6_datastream, 4)
        # self.report_result(
        #     self.n_char_before_sop,
        #     prefix="The number of characters in the datasream buffer before start-of-packet marker is"
        # )
        # self.n_char_before_som = self.count_chars_before_marker(self.d6_datastream, 14)
        # self.report_result(
        #     self.n_char_before_som,
        #     prefix="The number of characters in the datasream buffer before start-of-message marker is"
        # )
        self.d7_terminal_output = self.get_input_d7_terminal_output()
        self.dir_tree = self.build_directory_tree_from_temrinal_output(self.d7_terminal_output)
        self.cum_size_folder_below_1e4 = self.calc_cum_size_folders_in_dir_tree(self.dir_tree, 1e5)
        self.report_result(
            self.cum_size_folder_below_1e4,
            prefix="the cumulative size of folders with size <1e4 is "
        )
        self.smallest_dir_to_delete = self.find_smallest_dire_to_delete(self.dir_tree)
        self.report_result(
            self.smallest_dir_to_delete,
            prefix="the size of the smallest folder to delete to reach sufficent free space is"
        )
        self.report_end()

    def get_input_lol_of_calories(self) -> list:
        """"""
        input_file = self.inputs_dir / 'calory_count.txt'
        with open(input_file, 'r') as f:
            raw_data = f.read()

        calories_lol = [
            [int(n) for n in elf_inv.split('\n')] for elf_inv in raw_data.split('\n\n')
        ]
        return calories_lol

    def get_input_rucksacks_iventory(self):
        input_file = self.inputs_dir / 'rucksacks_inventories.txt'
        with open(input_file, 'r') as f:
            lines = [l.rstrip('\n') for l in f.readlines()]

        inventories = [[line[:int(len(line)/2)], line[int(len(line)/2):]]for line in lines]
        return inventories

    def get_input_cleaning_duty_sections_assignments(self):
        input_file = self.inputs_dir / 'cleaning_duty_pair_section_assignements.txt'
        with open(input_file, 'r') as f:
            lines = [l.rstrip('\n') for l in f.readlines()]

        pair_section_ranges  = []
        for line in lines:
            sas = line.split(',')
            pair_sas = []
            for sa in sas:
                sids = [int(s_str) for s_str in sa.split('-')]
                pair_sas.append(sids)
            pair_section_ranges.append(pair_sas)
        return pair_section_ranges

    def get_input_crates_rearrangement_planning(self):
        input_file = self.inputs_dir / 'crates_rearrangement_planning.txt'
        with open(input_file, 'r') as f:
            txt = f.read()

        stacks_raw, program_raw = [e.split('\n') for e in txt.split('\n\n')]

        # parse crate stacks definition
        # each line is splitted in segments of length 4
        ids = [c.strip() for c in re.split('\s+', stacks_raw[-1]) if c]
        stacks = {}
        for idx in ids:
            stacks[idx] = [line[(int(idx)-1)*4 + 1] for line in stacks_raw[-2::-1] if len(line) > (int(idx)-1)*4 and line[(int(idx)-1)*4 + 1] != ' ']

        # parse moves program
        moves = [re.search("move ([0-9]+) from ([0-9]) to ([0-9])", line).groups() for line in program_raw]

        return stacks, moves

    def get_input_d6_datasream_buffer(self) -> str:
        input_file = self.inputs_dir / 'd6_datastream_buffer.txt'
        with open(input_file, 'r') as f:
            data = f.read()

        return data

    def get_input_d7_terminal_output(self) -> list:
        input_file = self.inputs_dir / 'd7_terminal_output.txt'
        with open(input_file, 'r') as f:
            data = f.read()

        res = data.split('$')
        res = [[e.strip('\n').lstrip() for e in c.split('\n', 1)] for c in res if c != '']
        return res

    def find_max_of_calories(self, calories_lol):
        return max(sum(elf_calories) for elf_calories in calories_lol)

    def find_max_of_top_3_calories_providers(self, calories_lol):
        # sort lol of calories by sum
        sorted_calories_lol = sorted(calories_lol, key=lambda x: sum(x), reverse=True)

        return sum(sum(elem) for elem in sorted_calories_lol[:3])

    def find_priority_sum_of_misplaced_items(self, inventories):

        priority_sum = 0
        for lcomp, rcomp in inventories:
            # find common item type
            common_types = [l for l in lcomp if l in rcomp]

            if len(common_types) >= 1:
                common_type = common_types[0]
                # add priority
                priority_sum += string.ascii_lowercase.index(common_type.lower())+1 + (0 if common_type.lower() == common_type else 26)
            else:
                pass
        return priority_sum

    def find_priority_sum_of_badges_items(self, inventories):

        priority_sum = 0
        for group1, group2, group3 in self.grouper(inventories, 3):
            group1 = ''.join(group1)
            group2 = ''.join(group2)
            group3 = ''.join(group3)

            # find common item
            for cit in set(group1):
                if cit in set(group2):
                    if cit in set(group3):
                        break
            else:
                raise RuntimeError('No common item found')
            # add priority
            priority_sum += string.ascii_lowercase.index(cit.lower())+1 + (0 if cit.lower() == cit else 26)

        return priority_sum

    def find_cleaning_duty_redundant_assignments(self, psrs):
        """"""
        res = 0
        for psr in psrs:
            psa1, psa2 = psr[0], psr[1]

            if (psa1[0] >= psa2[0] and psa1[1] <= psa2[1] ) or \
                    (psa2[0] >= psa1[0] and psa2[1] <= psa1[1]):
                res += 1
        return res

    def find_cleaning_duty_overlapping_assignments(self, psrs):
        count = 0
        for psr in psrs:
            sa1, sa2 = psr[0], psr[1]
            if sa2[0]<= sa1[0] <= sa2[1] or sa1[0] <= sa2[0] <= sa1[1]:
                count += 1
        return count

    def find_top_crates_after_rearrangement(self, stacks, program, crane_model=9000):
        """"""
        new_stacks = copy.deepcopy(stacks)
        # apply rearrangement program
        for qty, origin, target in program:
            qty = int(qty)

            # extract qty from origin
            to_move, new_stacks[origin] = new_stacks[origin][-qty:], new_stacks[origin][:-qty]

            if crane_model == 9000:
                new_stacks[target] = new_stacks[target] + to_move[::-1]
            elif crane_model == 9001:
                new_stacks[target] = new_stacks[target] + to_move
            else:
                raise ValueError(f"Unknown crane model {crane_model}!")

        # collect names of top crates
        top_crates = ''
        for idx, stack in new_stacks.items():
            top_crates += stack[-1]

        return top_crates

    def build_directory_tree_from_temrinal_output(self, to_lst):
        """"""
        root = Folder('/', None)
        cwd = root
        for cmd, out in to_lst:

            # cd command
            if cmd.startswith('cd'):
                m = re.match('cd ([a-zA-Z\./]+)', cmd)
                dir_name = m.groups()[0]
                if all(c == '.' for c in dir_name):
                    cwd = cwd.parents[dir_name.count('.') -2]
                elif dir_name in cwd.subs_dict.keys():
                    cwd = cwd.subs_dict[dir_name]
                else:
                    assert dir_name == cwd.name

            # ls command
            elif cmd.startswith('ls'):
                contents_raw = [c for c in out.split('\n') if c!= '']
                for c in contents_raw:
                    if c.startswith('dir'):
                        c_dir_name = c.split(' ')[1].strip()
                        cwd.add_sub(Folder(name=c_dir_name, parent=cwd))
                    else:
                        size, c_file_name = c.split(' ')
                        cwd.add_sub(File(name=c_file_name, parent=cwd, size=int(size.strip())))

        return root

    def sort_directories_by_size(self, root: Folder):
        dirs_sizes = [(root.name, root.size)]

        for sub in root.subs:
            if isinstance(sub, Folder):
                dirs_sizes.extend(self.sort_directories_by_size(sub))

        return sorted(dirs_sizes, key=lambda x: x[1])

    def find_smallest_dire_to_delete(self, root):
        sorted_dirs_size = self.sort_directories_by_size(root)

        # find space to remove
        to_remove = root.size - 40e6

        res = min([d[1] for d in sorted_dirs_size if d[1] > to_remove])
        return res

    def calc_cum_size_folders_in_dir_tree(self, root, threshold):
        total = 0
        print(root.size)
        if root.size <= threshold:
            total += root.size

        if isinstance(root, Folder):
            for sub in root.subs:
                if isinstance(sub, Folder):
                    total += self.calc_cum_size_folders_in_dir_tree(sub, threshold)

        return total

    def count_chars_before_marker(self, dsb, marker_length):
        for idx in range(marker_length, len(dsb)):
            if len(set(dsb[idx-marker_length:idx]))==marker_length:
                break
        return idx

    def report_start(self):
        self.start = datetime.datetime.now()
        print(f'started the exploring at {self.start.strftime("%H:%M")}...')
        print('-'*100)

    def report_end(self):
        self.end = datetime.datetime.now()
        self.delay = self.end - self.start
        print('-'*100)
        print(f'finished the exploring at {self.end.strftime("%H:%M")} ({self.strfdelta(self.delay)})...')

    def report_results(self, results: list, msgs: list = None):
        for idx, res in enumerate(results):
            if msgs:
                self.report_result(res, msgs[idx])
            else:
                self.report_result(res, 'N/D')

    def report_result(self, res: any, msg_tmpl='', prefix='', suffix='', green_res=True):
        if green_res:
            res_str = self.green_str(str(res))
        else:
            res_str = str(res)

        if msg_tmpl:
            print(msg_tmpl.format(res_str))

        else:
            print(f'{prefix:<60} {res_str:<15} {suffix}')

    def report_exploration(self):
        print('='*50)
        for k,v in vars(self).items():
            if not k.startswith('_'):
                print(f'{k:<30}: {v}')
        print('=' * 50)

    @staticmethod
    def count_chars_in_string(txt: str) -> dict:
        chars = sorted(set(txt))
        counts =  {c: 0 for c in chars}

        for c in txt:
            counts[c] +=1
        return counts

    @staticmethod
    def _get_extreme_bit_at_position(idx: int, bits: list, mode):
        nbits = len(bits)

        vbit = ''.join(b[idx] for b in bits)
        count0 = vbit.count('0')
        if count0 > nbits / 2:
            if mode == 'most':
                return '0'
            else:
                return '1'
        elif count0 < nbits / 2:
            if mode == 'most':
                return '1'
            else:
                return '0'
        else:
            if mode == 'most':
                return '1'
            else:
                return '0'

    @staticmethod
    def green_str(msg):
        return Fore.GREEN + msg + Fore.RESET

    @staticmethod
    def grouper(iterable, n, *, incomplete='fill', fillvalue=None):
        """Collect data into non-overlapping fixed-length chunks or blocks"""
        # grouper('ABCDEFG', 3, fillvalue='x') --> ABC DEF Gxx
        # grouper('ABCDEFG', 3, incomplete='strict') --> ABC DEF ValueError
        # grouper('ABCDEFG', 3, incomplete='ignore') --> ABC DEF
        args = [iter(iterable)] * n
        if incomplete == 'fill':
            return zip_longest(*args, fillvalue=fillvalue)
        if incomplete == 'strict':
            return zip(*args, strict=True)
        if incomplete == 'ignore':
            return zip(*args)
        else:
            raise ValueError('Expected fill, strict, or ignore')

    def strfdelta(self, tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02.0f}s {mS:03.0f}ms', inputtype='timedelta'):
        """Convert a datetime.timedelta object or a regular number to a custom-
        formatted string, just like the stftime() method does for datetime.datetime
        objects.

        The fmt argument allows custom formatting to be specified.  Fields can
        include seconds, minutes, hours, days, and weeks.  Each field is optional.

        Some examples:
            '{D:02}d {H:02}h {M:02}m {S:02.0f}s' --> '05d 08h 04m 02s' (default)
            '{W}w {D}d {H}:{M:02}:{S:02.0f}'     --> '4w 5d 8:04:02'
            '{D:2}d {H:2}:{M:02}:{S:02.0f}'      --> ' 5d  8:04:02'
            '{H}h {S:.0f}s'                       --> '72h 800s'

        The inputtype argument allows tdelta to be a regular number instead of the
        default, which is a datetime.timedelta object.  Valid inputtype strings:
            's', 'seconds',
            'm', 'minutes',
            'h', 'hours',
            'd', 'days',
            'w', 'weeks'
        """

        # Convert tdelta to integer seconds.
        if inputtype == 'timedelta':
            remainder = tdelta.total_seconds()
        elif inputtype in ['s', 'seconds']:
            remainder = float(tdelta)
        elif inputtype in ['m', 'minutes']:
            remainder = float(tdelta) * 60
        elif inputtype in ['h', 'hours']:
            remainder = float(tdelta) * 3600
        elif inputtype in ['d', 'days']:
            remainder = float(tdelta) * 86400
        elif inputtype in ['w', 'weeks']:
            remainder = float(tdelta) * 604800

        f = Formatter()
        desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
        possible_fields = ('Y', 'm', 'W', 'D', 'H', 'M', 'S', 'mS', 'µS')
        constants = {'Y': 86400 * 365.24, 'm': 86400 * 30.44, 'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1,
                     'mS': 1 / pow(10, 3), 'µS': 1 / pow(10, 6)}
        values = {}
        for field in possible_fields:
            if field in desired_fields and field in constants:
                Quotient, remainder = divmod(remainder, constants[field])
                values[field] = int(Quotient) if field != 'S' else Quotient + remainder
        return f.format(fmt, **values)


if __name__ == '__main__':
    me = ExpeditionLeader()
    me.do_some_exploring()
