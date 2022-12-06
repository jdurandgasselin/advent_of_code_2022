from pathlib import Path
from typing import Union
import re
import pandas as pd
import numpy as np
from copy import deepcopy
from typing import List
from itertools import product
from collections import Counter
import bisect
import time
import queue
import seaborn as sns

class SubmarinePilot:

    def __init__(self):
        """"""
        self.inputs_dir = Path(__file__).parent / 'inputs'
        print('pilot created...')

    def do_some_piloting(self):
        print('started the piloting...')
        # self.depths = self.get_input_depths()
        # self.commands = self.get_input_commands()
        # self.diagnostic = self.get_input_diagnostic()
        # self.draws, self._boards = self.get_input_bingo_data()
        # self.df_tv = self.get_input_thermal_vents_data()
        # self.fish_population_0 = self.get_input_fish_population()
        # self.crab_positions = self.get_input_crab_positions()
        # self.signals_observations = self.get_input_signals_observations()
        # self.heatmap = self.get_input_heatmap()
        # self.code_lines = self.get_input_code()
        # self.oct_energy_map = self.get_input_octopus_energy_map()
        # self.caves_connections = self.get_input_caves_map()
        # self.dots, self.folds = self.get_input_page_1()
        # self.polymer_tmpl, self.polymer_insertions = self.get_input_polymer_data()
        self.chiton_dm = self.get_input_chiton_density_map()

        # self.depths_increase_1elem = self.count_element_increase_from_list(self.depths)
        # self.depths_sums_3elem = self.sum_3_consecutive_elements_from_list(self.depths)
        # self.depths_increase_3elem = self.count_element_increase_from_list(self.depths_sums_3elem)
        # self.df_motion = self.calulate_2d_motion((0, 0), self.commands)
        # self.final_x, self.final_z =  self.df_motion.loc[self.df_motion.index[-1], ['xa', 'za']]
        # self.xf_times_zf = self.final_x * self.final_z
        # self.gamma_str, self.epsilon_str = self.get_power_rates_from_diagnostic(self.diagnostic)
        # self.gamma, self.epsilon = int(self.gamma_str, 2), int(self.epsilon_str, 2)
        # self.power_consuption = self.gamma * self.epsilon
        #
        # self.oxygen_str, self.co2_str = self.get_life_support_rates(self.diagnostic)
        # self.oxygen, self.co2 = int(self.oxygen_str, 2), int(self.co2_str, 2)
        # self.life_support_rating = self.oxygen * self.co2
        # self.bingo_winning_board, self.bingo_winner_score = self.win_bingo(self.draws, self._boards)
        # self.bingo_losing_board, self.bingo_loser_score = self.lose_bingo(self.draws, self._boards)
        # self.count_multi_vents_points = self.avoid_thermal_vents(self.df_tv)
        # self.count_multi_vents_points_better = self.avoid_thermal_vents_but_better(self.df_tv)
        # self.fish_population_80 = self.predict_fish_population(self.fish_population_0, 80)
        # self.fish_population_256 = self.predict_fish_population_but_better(self.fish_population_0, 256)
        # self.crab_fuel_costs = self.calculate_crab_alignment_fuel_costs(self.crab_positions)
        # self.crab_fuel_costs_crab_way = self.calculate_crab_alignment_fuel_costs_the_crab_way(self.crab_positions)
        # self.number_1478_digits = self.count_1478_digits(self.signals_observations)
        # self.sum_of_decoded_numbers = self.decode_signal(self.signals_observations)
        # heatmap_low_points = self.identify_heatmap_low_points(self.heatmap)
        # self.heatmap_risk = self.calculate_heatmap_risk(heatmap_low_points)
        # self.prod_surf_3_largest_basins = self.calculate_size_of_heatmap_basins(self.heatmap, heatmap_low_points)
        # self.syntax_error_score = self.calculate_syntax_error_code(self.code_lines)
        # self.middle_incomplete_score = self.calculate_middle_incomplete_score(self.code_lines)
        # self.n_flahses = self.simulate_octopus_100steps(self.oct_energy_map)
        # self.sync_flash_step = self.find_synchronized_flash(self.oct_energy_map)
        # self.n_caves_pathways = self.find_all_caves_pathways(self.caves_connections)
        # self.n_caves_pathways_v2 = self.find_all_caves_pathways_the_longer_way(self.caves_connections)
        # self.ndots_fold1 = self.fold_manual_page_1(self.dots, self.folds)
        # self.polymer_rating_10steps = self.build_polymer_from_instructions(self.polymer_tmpl, self.polymer_insertions, 10)
        # self.polymer_rating_40steps = self.build_polymer_from_instructions(self.polymer_tmpl, self.polymer_insertions, 40)
        self.chiton_dm_large = self.build_large_cave_system(self.chiton_dm)
        self.lowest_risk = self.find_lowest_risk_path(self.chiton_dm_large)
        print('finished the piloting.')
        self.report()

    def report(self):
        print('='*50)
        for k,v in vars(self).items():
            if not k.startswith('_'):
                print(f'{k:<30}: {v}')
        print('=' * 50)

    def get_input_depths(self, depths_file=None):
        depths_file = depths_file or self.inputs_dir / 'depths.txt'
        with open(depths_file, 'r') as f:
            depths = [int(l.strip()) for l in f.readlines()]

        return depths

    def get_input_diagnostic(self, diagnostic_file=None):
        diagnostic_file = diagnostic_file or self.inputs_dir / 'binary_diagnostic.txt'
        with open(diagnostic_file, 'r') as f:
            diagnostic = [l.strip() for l in f.readlines()]

        return diagnostic

    def get_input_commands(self, commands_file=None):
        commands_file = commands_file or self.inputs_dir / 'commands.txt'
        with open(commands_file, 'r') as f:
            commands_strs = [l.strip() for l in f.readlines()]

        commands = []
        for cstr in commands_strs:
            m = re.match(r'(\w+)\s(\d+)', cstr)
            commands.append((m[1], int(m[2])))
        return commands

    def get_input_bingo_data(self, bingo_file=None):
        bingo_file = bingo_file or self.inputs_dir / 'bingo_data.txt'

        with open(bingo_file, 'r') as f:
            draws_str = f.readline().strip()
            boards_str = f.read().strip()

        boards = []
        boards_str_lst = boards_str.split('\n\n')
        for bstr in boards_str_lst:
            lines = bstr.split('\n')
            lines_digits = [[int(d) for d in re.split('\s+', l.strip())] for l in lines]
            boards.append(pd.DataFrame(index=range(len(lines)),
                                       columns=range(len(lines_digits[0])),
                                       data=lines_digits
                                       ))

        draws = [int(d) for d in draws_str.split(',')]
        return draws, boards

    def get_input_thermal_vents_data(self, thermo_vents_file=None):
        thermo_vents_file = thermo_vents_file or self.inputs_dir / 'thermal_vents.txt'

        with open(thermo_vents_file, 'r') as f:
            tv_lines_str = f.readlines()

        tv_pattern = re.compile(r'(\d+),(\d+)\s?->\s?(\d+),(\d+)')
        df_tv = pd.DataFrame(index=range(len(tv_lines_str)), columns=['x1','y1', 'x2', 'y2'])
        for i, tv_str in enumerate(tv_lines_str):
            m = re.match(tv_pattern, tv_str.strip())
            if m is not None:
                df_tv.loc[i, :] = [int(c) for c in m.groups()]

        return df_tv

    def get_input_fish_population(self, fish_population_file=None):
        fish_population_file = fish_population_file or self.inputs_dir / 'lantern_fish_population.txt'

        with open(fish_population_file, 'r') as f:
            return [int(d) for d in f.readline().split(',')]

    def get_input_crab_positions(self, crab_positions_file=None):
        crab_positions_file = crab_positions_file or self.inputs_dir / 'crab_positions.txt'
        with open(crab_positions_file, 'r') as f:
            crab_positions = [int(p) for p in f.read().strip().split(',')]

        return crab_positions

    def get_input_signals_observations(self, signals_observations_file=None):
        signals_observations_file = signals_observations_file or self.inputs_dir / 'signals_observations.txt'
        with open(signals_observations_file, 'r') as f:
            observations = [tuple(l.split(' | ')) for l in f.readlines()]

            data = []
            for tud, dd in observations:
                data.append((tud.strip().split(' '), dd.strip().split(' ')))

        return data

    def get_input_heatmap(self, heatmap_file=None):
        heatmap_file = heatmap_file or self.inputs_dir / 'heatmap.txt'
        with open(heatmap_file, 'r') as f:
            data = [l.strip() for l in f.readlines()]

            r,c = len(data), len(data[0])
            df = pd.DataFrame(index=list(range(r)), columns=list(range(c)),
                              data=[[int(d) for d in list(l)] for l in data])

        return df

    def get_input_code(self, code_file=None):
        code_file = code_file or self.inputs_dir / 'code_file.txt'
        with open(code_file, 'r') as f:
            return [ l.strip() for l in f.readlines()]

    def get_input_octopus_energy_map(self, octopus_energy_map_file=None):
        octopus_energy_map_file = octopus_energy_map_file or self.inputs_dir / 'octopus_energy_map.txt'
        with open(octopus_energy_map_file, 'r') as f:
            data = [l.strip() for l in f.readlines()]

            r, c = len(data), len(data[0])
            df = pd.DataFrame(index=list(range(r)), columns=list(range(c)),
                              data=[[int(d) for d in list(l)] for l in data])

        return df

    def get_input_caves_map(self, caves_map_file=None):
        caves_map_file = caves_map_file or self.inputs_dir / 'caves_map.txt'
        with open(caves_map_file, 'r') as f:
            return [l.strip() for l in f.readlines()]

    def get_input_page_1(self, page_1_file=None):
        page_1_file = page_1_file or self.inputs_dir / 'input_page_1.txt'
        with open(page_1_file, 'r') as f:
            data = f.read()

        fold_pattern  = r'fold along (x|y)=([0-9]+)'
        data2 = data.split('\n\n')
        dots = [ (int(l.split(',')[0]), int(l.split(',')[1])) for l in data2[0].split('\n')]

        folds = []
        for l in data2[1].split('\n'):
            m = re.search(fold_pattern, l)
            folds.append((m[1], int(m[2])))


        return dots, folds

    def get_input_polymer_data(self, polymer_data_file=None):
        polymer_data_file = polymer_data_file or self.inputs_dir / 'polymer_schematics.txt'
        with open(polymer_data_file, 'r') as f:
            data = f.read()

        fold_pattern = r'([A-Z]{2}) -> ([A-Z])'
        data2 = data.split('\n\n')
        template = data2[0]

        insertions = []
        for l in data2[1].split('\n'):
            m = re.search(fold_pattern, l)
            insertions.append((m[1], m[2]))


        return template, insertions

    def get_input_chiton_density_map(self, chiton_density_map_file=None):
        chiton_density_map_file = chiton_density_map_file or self.inputs_dir / 'chiton_density_map.txt'
        with open(chiton_density_map_file, 'r') as f:
            df = pd.DataFrame(data=[[int(d) for d in l.strip()] for l in f.readlines()])

        return df

    def count_element_increase_from_list(self, lst: list):
        result = 0
        for i in range(1, len(lst)):
            if lst[i] > lst[i - 1]:
                result += 1
        return result

    def sum_3_consecutive_elements_from_list(self, lst:list):
        return [sum(lst[i - 2:i + 1]) for i in range(2, len(lst))]

    def calulate_2d_motion(self, start: tuple, commands: list):
        """
        Args:
            start (tuple): aim, x, z
            commands (list): list of command tuples
        Returns:
            pd.DataFrame
        """

        df = pd.DataFrame(index=range(len(commands)), columns=['ab', 'xb', 'zb', 'cmd_dir', 'cmd_mg', 'da', 'dx', 'dz', 'aa', 'xa', 'za'])
        df.loc[0, ['ab', 'xb', 'zb']] = (0, 0, 0)

        df[['cmd_dir', 'cmd_mg']] = commands

        for i in df.index:
            if df.loc[i, 'cmd_dir'] == 'down':
                df.loc[i, 'da'] = df.loc[i, 'cmd_mg']
                df.loc[i, 'dx'] = 0
                df.loc[i, 'dz'] = 0

            elif df.loc[i, 'cmd_dir'] == 'up':
                df.loc[i, 'da'] = - df.loc[i, 'cmd_mg']
                df.loc[i, 'dx'] = 0
                df.loc[i, 'dz'] = 0
            else:
                df.loc[i, 'da'] = 0
                df.loc[i, 'dx'] = df.loc[i, 'cmd_mg']
                df.loc[i, 'dz'] = df.loc[i, 'ab'] * df.loc[i, 'cmd_mg']

            df.loc[i, ['aa', 'xa', 'za']] = df.loc[i, ['ab', 'xb', 'zb']].array + df.loc[i, ['da', 'dx', 'dz']].array
            if i < len(commands)-1:
                df.loc[i+1, ['ab', 'xb', 'zb']] = df.loc[i, ['aa', 'xa', 'za']].array
        return df

    def get_power_rates_from_diagnostic(self, diagnostic: list):
        # length of the diagnostic bits
        lbit = len(diagnostic[0])

        # gamma for most frequent, eppsilon for least fequent
        gamma, epsilon = '', ''
        for i in range(lbit):
                gamma += self._get_extreme_bit_at_position(i, diagnostic, 'most')
                epsilon += self._get_extreme_bit_at_position(i, diagnostic, 'least')

        return gamma, epsilon

    def get_life_support_rates(self, diagnostic):

        lbit = len(diagnostic[0])
        obits, cbits = deepcopy(diagnostic), deepcopy(diagnostic)

        for i in range(lbit):
            if len(obits) > 1:
                obits = [b for b in obits if b[i] == self._get_extreme_bit_at_position(i, obits, 'most')]
            if len(cbits) > 1:
                cbits = [b for b in cbits if b[i] == self._get_extreme_bit_at_position(i, cbits, 'least')]

        assert len(obits) == 1
        assert len(cbits) == 1

        return obits[0], cbits[0]

    def win_bingo(self, draws: List[int], boards: List[pd.DataFrame]):
        """"""
        print('winning at bingo...')
        game_over, winners = False, []
        for di, draw in enumerate(draws):
            for board in boards:
                board.replace(draw, pd.NA, inplace=True)
                marked = board.isna()
                marked_cols, marked_rows = marked.all(axis='index'),marked.all(axis='columns')
                if marked_cols.any() or marked_rows.any():
                    game_over = True
                    winners.append(board)

            if game_over:
                print(f'won at bingo after {di} draws.')
                print('fuck the squid!!')
                break

        if len(winners) > 1:
            raise RuntimeError('more than 1 winners for bingo!')
        else:
            board = winners[0]
            score = board.fillna(0).values.sum() * draw

        return board, score

    def lose_bingo(self, draws: List[int], boards: List[pd.DataFrame]):
        """"""
        print('Losing at bingo...')
        ranking = []
        for di,draw in enumerate(draws):
            winners = []
            for bi, board in enumerate(boards):
                # apply the draw
                board.replace(draw, pd.NA, inplace=True)

                # check if board wins
                marked = board.isna()
                marked_cols, marked_rows = marked.all(axis='index'),marked.all(axis='columns')
                if marked_cols.any() or marked_rows.any():
                    winners.append(bi)

            # remove boards that won this turn from boards still in play & add to ranking
            ranking.extend(boards[wbi] for wbi in winners)
            boards = [boards[i] for i in range(len(boards)) if i not in winners]

            if len(boards) == 0:
                print(f'all boards won at bingo after {di} draws.')
                break

        # identify board that won last, calculate score
        board = ranking[-1]
        score = board.fillna(0).values.sum() * draw

        return board, score

    def avoid_thermal_vents(self, df_tv:pd.DataFrame):
        """Evaluates the number of points where two or more thermal vents pass"""
        print('started avoiding thermal vents...')
        # size of terrain is determined by max of X & max of Y
        Xmax = int(df_tv[['x1', 'x2']].max(axis=0).max())
        Ymax = int(df_tv[['y1', 'y2']].max(axis=0).max())
        df_map = pd.DataFrame(data=np.zeros((Xmax+1, Ymax+1)), index=range(Ymax+1), columns=range(Xmax+1))
        # filter df to keep only rows with horizontal or vertical vents
        df_tv2 = df_tv[(df_tv['x2'] - df_tv['x1']) * (df_tv['y2'] - df_tv['y1']) == 0 ]

        for tvi, tv in df_tv2.iterrows():
            cells = product(range(min(tv.x1, tv.x2), max(tv.x1, tv.x2)+1), range(min(tv.y1, tv.y2), max(tv.y1, tv.y2)+1))
            for c in cells:
                df_map.loc[c[0], c[1]] += 1
            if tvi % 50 == 0:
                print(f'mapped {tvi} thermal vents...')

        count_points_mult_tv = (df_map >= 2).apply(np.count_nonzero).sum()
        print('avoided thermal vents')
        return count_points_mult_tv

    def avoid_thermal_vents_but_better(self, df_tv:pd.DataFrame):
        """Evaluates the number of points where two or more thermal vents pass, considering also diagonal vents"""
        print('started avoiding thermal vents but by not being stupid...')
        # size of terrain is determined by max of X & max of Y
        Xmax = int(df_tv[['x1', 'x2']].max(axis=0).max())
        Ymax = int(df_tv[['y1', 'y2']].max(axis=0).max())
        df_map = pd.DataFrame(data=np.zeros((Xmax+1, Ymax+1)), index=range(Ymax+1), columns=range(Xmax+1))

        for tvi, tv in df_tv.iterrows():
            # for horizontal or vertical vents, passing points are a product of coordinates
            if (tv.x2 - tv.x1) * (tv.y2 - tv.y1) == 0:
                cells = product(range(min(tv.x1, tv.x2), max(tv.x1, tv.x2)+1), range(min(tv.y1, tv.y2), max(tv.y1, tv.y2)+1))
            # for diagonal points
            else:
                dx = tv.x2 - tv.x1
                dy = tv.y2 - tv.y1
                assert abs(dx) == abs(dy)
                cells = [(tv.x1 + i * dx/abs(dx), tv.y1 + i * dy/abs(dy)) for i in range(0, abs(dx)+1)]

            for c in cells:
                df_map.loc[c[0], c[1]] += 1
            if tvi % 50 == 0:
                print(f'mapped {tvi} thermal vents...')

        count_points_mult_tv = (df_map >= 2).apply(np.count_nonzero).sum()
        print('avoided thermal vents like a boss')
        return count_points_mult_tv

    def predict_fish_population(self, population0: list, days_to_predict: int):
        """"""
        print('Started predicting fish population (wtf)...')

        # initialisation
        current_population = deepcopy(population0)

        # iteration
        for day in range(days_to_predict):
            next_population = []
            fish_spawn = 0
            for fish_spawn_delay in current_population:
                if fish_spawn_delay == 0:
                    fish_spawn += 1
                    next_population.append(6)
                else:
                    next_population.append(fish_spawn_delay - 1)

            next_population.extend([8 for i in range(fish_spawn)])
            current_population = next_population

            if day % 10 == 0:
                print(f'predicted {day} days of fish population growth (current population={len(current_population)})...')

        print('predicted fish population.')
        return len(current_population)

    def predict_fish_population_but_better(self, population0: list, days_to_predict: int):
        """"""
        print('Started predicting fish population (wtf)...')

        # initialisation
        current_population = np.array([Counter(population0).get(k, 0) for k in range(8, -1, -1)], dtype='int64')

        # iteration
        for day in range(days_to_predict):
            spawner_fish = current_population[-1]
            next_population = np.roll(current_population, 1)
            next_population[2] += spawner_fish

            current_population = next_population
            if day % 10 == 0:
                print(f'predicted {day} days of fish population growth...')

        print('predicted fish population.')
        return current_population.sum()

    def calculate_crab_alignment_fuel_costs(self, crabs_pos: list):
        """"""
        print('searching horizontal position that minimizes the crabs fuel costs...')

        median = np.median(crabs_pos)
        fuel_cost_min = sum([abs(v-median) for v in crabs_pos])
        print(f'Done searching (position={median}, cost={fuel_cost_min}).')
        return fuel_cost_min

    def calculate_crab_alignment_fuel_costs_the_crab_way(self, crabs_pos: list):
        """"""
        print('searching horizontal position that minimizes the crabs fuel costs...')

        pmvs = list(set(sorted(crabs_pos)))
        sums_consecutive_ints = [int(0.5*n*(n+1)) for n in range(max(pmvs)+1)]

        fuel_cost_min, target_pos = None, None

        for pmv in list(set(sorted(crabs_pos))):
            fuel_cost = 0
            for cp in crabs_pos:
                moves = abs(cp-pmv)
                fuel_cost += sums_consecutive_ints[moves]

            if fuel_cost_min is None or fuel_cost < fuel_cost_min:
                fuel_cost_min = fuel_cost
                target_pos = pmv
            else:
                continue
        print(f'Done searching (position={target_pos}, cost={fuel_cost_min}).')
        return fuel_cost_min

    def count_1478_digits(self, sobs: list):
        dcount = 0
        for tud, dd in sobs:
            for d in dd:
                if len(d) in [2, 3, 4, 7]:
                    dcount +=1
        return dcount

    def decode_signal(self, sobs: list):
        ddef = {
            '0': ['T', 'TL', 'TR',      'BL', 'BR', 'B'],
            '1': [           'TR',            'BR'     ],
            '2': ['T',       'TR', 'M', 'BL',       'B'],
            '3': ['T',       'TR', 'M',       'BR', 'B'],
            '4': [     'TL', 'TR', 'M',       'BR'     ],
            '5': ['T', 'TL',       'M', 'BR', 'B'],
            '6': ['T', 'TL',       'M', 'BR', 'BL', 'B'],
            '7': ['T',       'TR',      'BR'      ],
            '8': ['T', 'TL', 'TR', 'M', 'BL', 'BR', 'B'],
            '9': ['T', 'TL', 'TR', 'M',       'BR', 'B'],
        }
        numbers = []
        for signal, dd in sobs:

            # find signals corresponding to digits 1,4,7,8 since they have unique lengths
            txt1 = [s for s in signal if len(s)==len(ddef['1'])][0]
            txt4 = [s for s in signal if len(s)==len(ddef['4'])][0]
            txt7 = [s for s in signal if len(s)==len(ddef['7'])][0]
            txt8 = [s for s in signal if len(s)==len(ddef['8'])][0]

            # 1 contains TR & BR
            tr_br = list(txt1)
            assert len(tr_br) == 2
            # T is the only component of 7 not in 1
            t = [p for p in txt7 if p not in txt1][0]
            # TL & M are in 4 but not in 1
            tl_m = [p for p in txt4 if p not in txt1]
            assert len(tl_m) == 2 
            # B & BL are in 8 and are the only unknowns
            b_bl = [p for p in txt8 if p !=t and p not in tr_br and p not in tl_m]
            assert len(b_bl) == 2

            # remove signals for which the corresponding digit has been identified
            signal2 = [s for s in signal if s not in [txt1, txt4, txt7, txt8]]

            # in the remaining signals, 3, 2 and 5 have length =5
            s_235 = [s for s in signal2 if len(s)== 5]
            assert len(s_235) == 3
            # 3 is the one that contains tr and br
            s3 = [s for s in s_235 if all(c in s for c in tr_br)][0]
            # 5 is the one that contains tl and m
            s5 = [s for s in s_235 if all(c in s for c in tl_m)][0]
            tr = [c for c in s3 if c not in s5][0]
            tl = [c for c in s5 if c not in s3][0]
            br = [c for c in tr_br if c != tr][0]
            m = [c for c in tl_m if c != tl][0]

            # B is the unknown left in 5
            b = [c for c in s5 if c not in [t, tl, m, br]][0]
            bl = [c for c in b_bl if c != b][0]

            # deduce mapping
            mapp = {
                t: 'T',
                tl: 'TL',
                tr: 'TR',
                m: 'M',
                bl: 'BL',
                br: 'BR',
                b: 'B',
            }

            # decode digits display:
            digits = []
            for d in dd:
                chars = list(d)
                positions = set([mapp[c] for c in chars])
                digit = [k for k in ddef.keys() if set(ddef[k]) == positions]
                digits.append(digit[0])

            numbers.append(int(''.join(digits)))

        return sum(numbers)

    def calculate_heatmap_risk(self, hmlp: pd.DataFrame):
        """"""
        risked_heatmap = hmlp + 1
        risk = risked_heatmap.fillna(0).values.sum()
        return risk

    def identify_heatmap_low_points(self, hm: pd.DataFrame):
        greater_above = hm.diff(periods=1, axis=0).fillna(-1) < 0
        greater_below = hm.diff(periods=-1, axis=0).fillna(-1) < 0
        greater_left = hm.diff(periods=1, axis=1).fillna(-1) < 0
        greater_right = hm.diff(periods=-1, axis=1).fillna(-1)< 0

        hmlp = hm[greater_above & greater_below & greater_left & greater_right]
        return hmlp

    def calculate_size_of_heatmap_basins(self, hm: pd.DataFrame, hmlp: pd.DataFrame):
        basins_data = []
        for x in hmlp.columns:
            for y in hmlp.index:
                # non N/A values are low points
                if not pd.isna(hmlp.loc[y,x]):
                    # find basin for low point
                    basin = self.find_basin_from_low_point(x, y, hm)

                    # count size of basin
                    size = basin[basin != -1].count().sum()

                    # save it for sorting
                    basins_data.append((size, basin))

        basins_data = sorted(basins_data, key=lambda x: x[0], reverse=True)

        return basins_data[0][0] * basins_data[1][0] * basins_data[2][0]

    def find_basin_from_low_point(self, xlp: int, ylp: int, hm: pd.DataFrame):
        identified = [(ylp, xlp)]
        explored = []
        identified_but_not_explored = [t for t in identified if t not in explored]
        basin = pd.DataFrame(np.full(hm.shape, -1))

        xmax = max(hm.columns)
        xmin = min(hm.columns)
        ymax = max(hm.index)
        ymin = min(hm.index)

        # start the inspection of the basin
        while len(identified_but_not_explored) > 0:

            # find next point to explore:
            y, x = identified_but_not_explored[0]
            v = hm.loc[y, x]

            # exploration of point (y,x)
            if x > xmin and v <= hm.loc[y, x-1] < 9 and (y, x - 1) not in identified:
                identified.append((y, x-1))

            if x < xmax and v <= hm.loc[y, x + 1] < 9 and (y, x + 1) not in identified:
                identified.append((y, x + 1))

            if y > ymin and v <= hm.loc[y - 1, x] < 9 and (y - 1, x) not in identified:
                identified.append((y - 1, x))

            if y < ymax and v <= hm.loc[y + 1, x] < 9 and (y + 1, x) not in identified:
                identified.append((y + 1, x))

            # point is explored, i.e. its neighbours have been checked and then identified or ignored
            explored.append((y, x))
            basin.loc[y, x] = hm.loc[y, x]
            # updates list of points left to explore
            identified_but_not_explored = sorted([t for t in identified if t not in explored])

        return basin

    def calculate_syntax_error_code(self, code_lines: list):
        print('Checking navigation subsystem for syntax errors...')
        oc = ['(', '{', '[', '<']
        cc = [')', '}', ']', '>']

        oc2cc = {
            '(': ')',
            '{': '}',
            '[': ']',
            '<': '>',
        }

        cc2score = {
            ')': 3,
            '}': 1197,
            ']': 57,
            '>': 25137,
        }
        corrupting_characters = []
        for i, line in enumerate(code_lines):
            opened = []
            for c in line:
                # opening character
                if c in oc:
                    opened.append(c)

                # closing character
                elif c in cc:
                    # corrupted line
                    if c != oc2cc[opened[-1]]:
                        corrupting_characters.append(c)
                        break
                    else:
                        del opened[-1]

                else:
                    raise RuntimeError(f"Invalid character '{c}' found at line {i}")

        scores = [cc2score[c] for c in corrupting_characters]

        score = sum(scores)
        print(f'Finished checking navigation subsystem for syntax errors. score = {score}')

        return score

    def calculate_middle_incomplete_score(self, code_lines: list):
        print('Checking navigation subsystem for incomplete code lines...')
        oc = ['(', '{', '[', '<']
        cc = [')', '}', ']', '>']

        oc2cc = {
            '(': ')',
            '{': '}',
            '[': ']',
            '<': '>',
        }

        cc2score = {
            ')': 1,
            '}': 3,
            ']': 2,
            '>': 4,
        }
        scores = []
        for i, line in enumerate(code_lines):
            opened = []
            j = 0
            is_corrupted = False
            while not is_corrupted and j < len(line):
                c = line[j]
                # opening character
                if c in oc:
                    opened.append(c)

                # closing character
                elif c in cc:
                    # corrupted line
                    if c != oc2cc[opened[-1]]:
                        is_corrupted = True
                    else:
                        del opened[-1]

                else:
                    raise RuntimeError(f"Invalid character '{c}' found at line {i}")

                j += 1

            # incomplete lines
            if not is_corrupted and len(opened) > 0:
                score = 0
                ecc = [oc2cc[c] for c in reversed(opened)]
                for c in ecc:
                    score *= 5
                    score += cc2score[c]
                scores.append(score)

        total_score = np.median(sorted(scores))

        print(f'Finished checking navigation subsystem for incomplete code lines. score = {total_score}')

        return total_score

    def simulate_octopus_100steps(self, octopus_energy_map: pd.DataFrame):

        oem = deepcopy(octopus_energy_map)
        nf_total = 0

        for i in range(100):

            if i % 10 == 0:
                print(f"Simulating ocotpus life: step {i}/100...")
            # increase energy levels by 1
            oem = oem + 1
            mask = oem > 9
            # initial pass for octopus that are going to flash in this step
            flashing = self.get_positions(mask, True)
            # octopus that have flashed already in this step
            flashed = []

            to_flash = sorted([t for t in flashing if t not in flashed])
            while len(to_flash) > 0:

                y,x = to_flash[0]
                # flash current octopus
                oem.loc[(y-1 <= oem.index) & (oem.index <= y+1), (x-1 <= oem.columns) & (oem.columns <= x+1)] += 1
                # add current octopus to flashed
                flashed.append((y,x))

                # update flashing
                mask2  = oem > 9
                flashing = self.get_positions(mask2, True)

                # update to_flash
                to_flash = sorted([t for t in flashing if t not in flashed])

            # reset octopuses tha flashed to 0
            for t in flashed:
                oem.loc[t[0], t[1]] = 0

            # update total flash count
            nf_total += len(flashed)
            return nf_total

    def find_synchronized_flash(self, octopus_energy_map: pd.DataFrame):

        oem = deepcopy(octopus_energy_map)
        step = 1
        sync_flash = False
        while not sync_flash:
            if step % 10 == 0:
                print(f"looking for synchronized flash: step {step}/100...")
            # increase energy levels by 1
            oem = oem + 1
            mask = oem > 9
            # initial pass for octopus that are going to flash in this step
            flashing = self.get_positions(mask, True)
            # octopus that have flashed already in this step
            flashed = []

            to_flash = sorted([t for t in flashing if t not in flashed])
            while len(to_flash) > 0:

                y,x = to_flash[0]
                # flash current octopus
                oem.loc[(y-1 <= oem.index) & (oem.index <= y+1), (x-1 <= oem.columns) & (oem.columns <= x+1)] += 1
                # add current octopus to flashed
                flashed.append((y,x))

                # update flashing
                mask2  = oem > 9
                flashing = self.get_positions(mask2, True)

                # update to_flash
                to_flash = sorted([t for t in flashing if t not in flashed])

            # reset octopuses tha flashed to 0
            for t in flashed:
                oem.loc[t[0], t[1]] = 0

            # check for synchronized flash
            flash_mask = oem == 0
            if flash_mask.all(axis=None):
                sync_flash = True
                print(f"found synchronized flash at step {step}/100.")

            # otherwise increment step
            else:
                step += 1

        return step

    def find_all_caves_pathways_the_longer_way(self, map_conns: list):
        print('Exploring the cave system but slowly...')
        mcs = map_conns
        entries = [c for c in mcs if 'start' in c]
        exits = [c for c in mcs if 'end' in c]

        def co2t(conn: Union[str, list], first: str = None):
            def split_and_sort(c, f):
                t = c.split('-')
                assert len(t) == 2
                return sorted(t, key=lambda x: x==f, reverse=True)

            if isinstance(conn, list):
                if isinstance(first, list):
                    return [split_and_sort(ec, ef) for ec, ef in zip(conn, first)]
                else:
                    return [split_and_sort(ec, first) for ec in conn]
            else:
                return split_and_sort(conn, first)

        def find_conn(cave: str, first: str = None, split=True):
            a = [conn for conn in mcs if cave in conn]
            first = first or cave
            if split:
                return co2t(a, first)
            else:
                return a

        def is_big(cave: str):
            return cave.upper() == cave

        # initialize pathways with caves connected to start
        pws = find_conn('start')

        # exploration is completed when all pathways find the exit
        # each iteration creates new pathways from existing ones using connections to last position of
        # each pathway
        while not all([p[-1] == 'end' for p in pws]):

            # new pathways obtained for this iteration
            npws = []
            # iterate over pathways
            for cp in pws:
                # new pathways obtained from current pathways
                cnps = []
                # current position
                cpos = cp[-1]

                # ignore completed pathways
                if cpos == 'end':
                    cnps.append(cp)

                else:
                    # new connections
                    ncos = find_conn(cpos)
                    for nco in ncos:
                        # new cave
                        nca = nco[-1]
                        # small caves can be only visited 1 time and one small cave tow times
                        if not is_big(nca):
                            if nca == 'start':
                                is_ok = False
                            else:
                                # if a small cave has been visited twice, all other small caves can only be visited once
                                cp_sc = [c for c in cp if not is_big(c)]
                                if any(cp_sc.count(c) >= 2 for c in cp_sc):
                                    if nca in cp_sc:
                                        is_ok = False
                                    else:
                                        is_ok = True
                                else:
                                    is_ok = True

                        # otherwise create new pathways from existing one
                        else:
                            is_ok = True

                        if is_ok:
                            cnp = deepcopy(cp) + [nca]
                            cnps.append(cnp)

                # update new pathways of iteration with new pathways from current path
                npws.extend(cnps)

            # update pathways with newly obtained ones
            pws = npws

            print(f"Found {len(pws)} pathways...")

        n_pws = len(pws)
        print("Finished exploring the cave system but slowly.")
        return n_pws

    def find_all_caves_pathways(self, map_conns: list):

        mcs = map_conns
        entries = [c for c in mcs if 'start' in c]
        exits = [c for c in mcs if 'end' in c]

        def co2t(conn: Union[str, list], first: str = None):
            def split_and_sort(c, f):
                t = c.split('-')
                assert len(t) == 2
                return sorted(t, key=lambda x: x==f, reverse=True)

            if isinstance(conn, list):
                if isinstance(first, list):
                    return [split_and_sort(ec, ef) for ec, ef in zip(conn, first)]
                else:
                    return [split_and_sort(ec, first) for ec in conn]
            else:
                return split_and_sort(conn, first)

        def find_conn(cave: str, first: str = None, split=True):
            a = [conn for conn in mcs if cave in conn]
            first = first or cave
            if split:
                return co2t(a, first)
            else:
                return a

        def is_big(cave: str):
            return cave.upper() == cave

        # initialize pathways with caves connected to start
        pws = find_conn('start')

        # exploration is completed when all pathways find the exit
        # each iteration creates new pathways from existing ones using connections to last position of
        # each pathway
        while not all([p[-1] == 'end' for p in pws]):

            # new pathways obtained for this iteration
            npws = []
            # iterate over pathways
            for cp in pws:
                # new pathways obtained from current pathways
                cnps = []
                # current position
                cpos = cp[-1]

                # ignore completed pathways
                if cpos == 'end':
                    cnps.append(cp)

                else:
                    # new connections
                    ncos = find_conn(cpos)
                    for nco in ncos:
                        # new cave
                        nca = nco[-1]
                        # skip connections for which the new cave is small and already visited
                        if not is_big(nca) and nca in cp:
                            pass
                        # otherwise create new pathways from exsiting one
                        else:
                            cnp = deepcopy(cp) + [nca]
                            cnps.append(cnp)

                # update new pathways of iteration with new pathways from current path
                npws.extend(cnps)

            # update pathways with newly obtained ones
            pws = npws

        n_pws = len(pws)
        return n_pws

    def get_positions(self, df: pd.DataFrame, value):
        """ Get index positions of value in dataframe i.e. dfObj."""
        res = list()
        # Get bool dataframe with True at positions where the given value exists
        mask = df.isin([value])
        # Get list of columns that contains the value
        seriesObj = mask.any()
        cnames = list(seriesObj[seriesObj == True].index)
        # Iterate over list of columns and fetch the rows indexes where value exists
        for col in cnames:
            rows = list(mask[col][mask[col] == True].index)
            for row in rows:
                res.append((row, col))
        # Return a list of tuples indicating the positions of value in the dataframe
        return res

    def fold_manual_page_1(self, dots: list, folds: list):
        print("Started folding the f** paper")
        xmax = max(d[0] for d in dots)
        ymax = max(d[1] for d in dots)
        page = pd.DataFrame(0, index=range(ymax+1), columns=range(xmax+1))

        # place dots on page
        for dot in dots:
            x, y = dot
            page.loc[y, x] = 1

        # fold page as indicated
        # horizontal folds from bottom to top
        # vertical folds from right to left
        for idx, fold in enumerate(folds):
            nh = page[page > 0].count().sum()

            direction, s = fold
            if direction == 'x':
                new_page = page.loc[:, :s-1].add(page.loc[:, :s+1:-1].values)

            else:
                new_page = page.loc[:s-1, :].add(page.loc[:s+1:-1, :].values)

            print(f'Fold {idx+1}: {direction}={s}. ({page.shape}) --> ({new_page.shape}).')
            page = new_page

        nh = page[page > 0].count().sum()
        print("Folded the f** paper.")
        return nh

    def build_polymer_from_instructions(self, tmpl: str, insertions: list, nsteps: int):
        print('Started building polymer from instructions...')

        ns, nsmax = 1, nsteps
        pol = deepcopy(tmpl)
        ins = {t[0]: t[0][0] + t[1] + t[0][-1] for t in insertions}
        counts = {}
        # initial counts of pairs
        for i in range(len(pol)-1):
            pair = pol[i:i+2]
            if pair in counts.keys():
                counts[pair] +=1
            else:
                counts[pair] = 1

        while ns <= nsmax:
            new_count = {}
            for pair, count in counts.items():
                if pair in ins.keys():
                    nt = ins[pair]
                    np1 = nt[0:2]
                    np2 = nt[1:]
                    for np in [np1, np2]:
                        if np in new_count.keys():
                            new_count[np] += count
                        else:
                            new_count[np] = count
                else:
                    new_count[pair] = count


            print(f'Fabricated {ns}/{nsmax} steps of polymer')
            # update the step number
            counts = deepcopy(new_count)
            ns += 1

        letters = {}
        for pair, ct in counts.items():
            for c in pair:
                letters[c] = letters[c] + ct if c in letters else ct

        letters['B'] += 1
        letters['N'] += 1

        countsv = [c/2 for c in letters.values()]
        cmin, cmax = min(countsv), max(countsv)

        print('Finished building the polymer.')
        return cmax - cmin

    def build_large_cave_system(self, cdm: pd.DataFrame):
        cdm2 = pd.DataFrame(data=np.zeros(tuple(s*5 for s in cdm.shape)))

        perm = {
            1:2,
            2:3,
            3:4,
            4:5,
            5:6,
            6:7,
            7:8,
            8:9,
            9:1
        }

        def increase_vals(x: int, n_permutations=0) -> int:
            res = x
            for i in range(n_permutations):
                res = perm[res]
            return res

        for i in range(5):
            for j in range(5):
                new = cdm.applymap(increase_vals, n_permutations=i+j)
                cdm2.loc[100*i:100*(i+1)-1, 100*j:100*(j+1)-1] = new.values

        return cdm2

    def find_lowest_risk_path(self, cdm: pd.DataFrame):
        print('Searching path with lowest risk...')
        # start and end nodes
        start = (0, 0)
        imin, cmin = min(cdm.index), min(cdm.columns)
        imax, cmax = max(cdm.index), max(cdm.columns)
        end = (imax, cmax)

        def heuristic(pos: tuple) -> int:
            return 3.5 * ((end[0] - pos[0]) + (end[1] - pos[1]))

        def find_neighbours(pos: tuple) -> list:
            # neighbours of current node
            cnes = []
            if pos[0] > imin:
                cnes.append((pos[0] - 1, pos[1]))
            if pos[0] < imax:
                cnes.append((pos[0] + 1, pos[1]))
            if pos[1] > cmin:
                cnes.append((pos[0], pos[1] - 1))
            if pos[1] < cmax:
                cnes.append((pos[0], pos[1] + 1))

            return cnes

        def disp_iter(q=None, res=None):
            if res is None:
                res = pd.DataFrame(np.zeros(cdm.shape))
                for p in q:
                    res.loc[p] = 1
            return res
        # costs & parents array
        # costs = cdm * np.inf
        # costs.loc[start] = 0
        parents = pd.DataFrame(np.full(cdm.shape, None))

        ct = 0  # step counter
        found = False  # loop stop switch
        # queue = PrioritizedQueue({pos: costs.loc[pos] for pos in list(product(cdm.index, cdm.columns))})
        pq = PrioritizedQueue()
        pq.add((start, 0, 0))
        visited = pd.DataFrame(np.full(cdm.shape, False))

        t1 = time.perf_counter()
        while not pq.empty() and not found:
            cc, cno = pq.get()

            if cno == end:
                t2 = time.perf_counter()
                found = True

            else:
                cnes = find_neighbours(cno)
                # update of cost for each neighbour of current node
                for cne in cnes:
                    # already visited nodes are ignored
                    # nodes with shorter paths are ignored
                    if visited.loc[cne]:
                        pass
                    else:
                        alt = cc + cdm.loc[cne]
                        heur = alt + heuristic(cne)
                        isin = pq.isin(cne)
                        if isin and alt < pq.cost(cne):
                            pq.update(cne, alt, heur)
                            parents.loc[cne] = cno

                        elif not isin:
                            pq.add(cne, alt, heur)
                            parents.loc[cne] = cno

                # add current node to closed queue
                visited.loc[cno] = True

            ct += 1
            if ct % 500 == 0:
                print(f'Explored {ct} nodes...')

        if len(pq) == 0:
            raise RuntimeError('Exhausted possible paths')

        print(f'Found shortest path in {ct} iterations')
        # reconstruct path
        path = []
        path2D = pd.DataFrame(np.full(cdm.shape, -10))
        cn = end
        while cn is not None:
            path.insert(0, cn)
            path2D.loc[cn] = cdm.loc[cn]
            cn = parents.loc[cn]

        risk = sum(cdm.loc[p] for p in path[1:])
        print(f'Found path with lowest risk = {risk} ({t2-t1} s)')
        return risk


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


class PrioritizedQueue:

    def __init__(self, pos2val: dict = None, pos2heur: dict = None):
        self.pos2val = pos2val or {}
        self.pos2heur = pos2heur or {}
        self.keysort = lambda x: self.pos2heur[x]
        self.dfin = pd.DataFrame(np.full((500,500), False))
        self.queue = sorted(list(self.pos2heur.keys()), key=self.keysort)

    def add(self, pos, val, heur):
        assert not self.dfin.loc[pos], f"{pos} already in queue"
        self.pos2val[pos] = val
        self.pos2heur[pos] = heur
        self.queue.insert(bisect.bisect_left(self.queue, heur, key=self.keysort), pos)

    def update(self, pos, val, heur):
        if self.dfin.loc[pos]:
            self.queue.remove(pos)
            self.dfin.loc[pos] = False

        self.add(pos, val, heur)

    def extract_min(self):
        res = self.queue[0]
        self.queue.remove(res)
        self.dfin.loc[res] = False
        return res

    def __len__(self):
        return len(self.queue)

    def isin(self, pos: tuple) -> bool:
        return self.dfin.loc[pos]

    def cost(self, pos):
        return self.pos2val[pos]

    def heur(self, pos):
        return self.pos2heur[pos]


if __name__ == '__main__':
    somebody = SubmarinePilot()
    somebody.do_some_piloting()