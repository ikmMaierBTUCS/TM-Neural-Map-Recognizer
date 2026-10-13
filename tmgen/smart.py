# Smart generator: verbatim copy of "TM MapGen(1).py" from
# https://github.com/ikmMaierBTUCS/Terra-Mystica-Map-Generator, branch "3"
# (commit 202b2f5e278ca1e125b5cbc67ce05a0a880cbffd), lines 1-602.
# Changes: IPython/matplotlib imports removed (only needed for display/plotting),
# the example/main loop at the end of the file removed.
import math
import random
import numpy as np

class HexagonalMap:
    def __init__(self, height, width, form, rivers, start = 0):
        # form = 0 => sharp, form = 1 => round
        self.width = width
        self.height = height
        self.form = form
        self.rivers = rivers
        self.map = {}
        # initialize map and set rivers
        for row in range(self.height):
            for col in range(self.width - ((row + self.form) % 2)):
                x = col
                y = row
                if (x,y) in self.rivers:
                    self.map[(x, y)] = ' ~ '
                else:
                    self.map[(x, y)] = '???'
        self.colors = ['red','yel','bro','bla','blu','grn','gry','~~~','???']
        self.luck = np.array([0]*7, dtype=np.float64)
        if start == 0:
            self.start = (random.randint(2,width - 3),random.randint(2,height - 3))
        else:
            self.start = start
            
    def reset(self):
        return HexagonalMap(self.height, self.width, self.form, self.rivers, self.start)
        
    def out_of_bounds(self,hexf):
        # is the hex outside of the map?
        if hexf[1] not in range(self.height) or hexf[0] not in range(self.width - ((hexf[1] + self.form) % 2)):
            return True
        else:
            return False
    
    def center_hex(self):
        middle_height = (self.height - 1) / 2
        if self.height % 2 == 0:
            middle_width = (self.width - 0.5) / 2
        elif self.height % 4 == 1:
            middle_width = (self.width - 1 - self.form) / 2
        elif self.height % 4 == 3:
            middle_width = (self.width - 2 + self.form) / 2
        return (middle_width,middle_height)
    
    def fly_distance(self,hex1,hex2):
        dist = abs(hex1[1] - hex2[1]) + max([0, abs(hex1[0] - hex2[0]) - abs(hex1[1] - hex2[1])/2])
        return dist
    
    def absolute_stock(self,color):
        # number of hexes of colour color that is yet on the map
        return list(self.map.values()).count(color)
    
    def value_of_hex(self,hexf):
        if self.out_of_bounds(hexf):
            return 0
        else:
            distance = self.fly_distance(hexf,self.center_hex())
            max_dist = self.width / 3 + self.height / 2
            return 1.5 - (distance / max_dist)**2
            
    def stock(self,color):
        # sum of weights of hexes of colour color that are yet on the map
        stock = 0
        for hexf in self.map.keys():
            if self.map[hexf] == color:
                stock += self.value_of_hex(hexf)
        return stock
        #return list(self.map.values()).count(color)
        
    def work_distance(self, hexf, color):
        # measure of work required to get from hexf to the next hex of colour color
        maximal_work = 168
        if self.stock(color) == 0: # check if any hexes of same color exist yet
            #print('no '+color+' yet')
            return maximal_work
        for direction in range(6): # check if color is adjacent
            if not self.out_of_bounds(self.next_hex(hexf,direction)) and self.map[self.next_hex(hexf,direction)] == color: 
                #print(color+' is adjacent')
                return 0
        '''for direction in range(6): # check if color is bridgable
            if self.map[self.next_hex(hexf,direction)] == 'riv' and self.map[self.next_hex(hexf,direction + 1)] == 'riv' and  self.map[self.next_hex(next_hex(hexf,direction),direction + 1)] == color: 
                #print(color+' is bridgable')
                return 5'''
        work = maximal_work
        for path_length in range(2,5):
            if work < 42*path_length:
                break
            path = path_length * [-1]
            for initial_direction in range(6):
                path[0] = initial_direction
                for further_steers in range(3**(path_length - 1)):
                    for further_step in range(1, path_length):
                        path[further_step] = ((further_steers // (3**(further_step - 1)) % 3) - 1 + path[further_step - 1]) % 6
                    # each possible path set up
                    #print('path :'+str(path))
                    transit_colors = []
                    path_legal = True
                    hex_in_path = hexf
                    for direction in path:
                        hex_in_path = self.next_hex(hex_in_path,direction)
                        if self.out_of_bounds(hex_in_path):
                            path_legal = False
                            break
                        else:
                            transcol = self.map[hex_in_path]
                            if transcol == ' ~ ':
                                transit_colors += ['~~~']
                            else:
                                transit_colors += [transcol]
                    #print('transit colors: '+str(transit_colors))
                    if path_legal and transit_colors[-1] == color:
                        path_work = path_cost(transit_colors)
                        if path_work < work:
                            #print(str([dir_str(dire) for dire in path]) + ' is shorter at '+str(path_work))
                            work = path_work
        return work
    
    def number_of_landhexes(self):
        actual_rivers = [riv for riv in self.rivers if not self.out_of_bounds(riv)]
        all_hexes = self.height * (self.width) - abs((self.height + self.form) // 2)
        return all_hexes - len(actual_rivers)
    
    def contingent(self, color):
        start_contingent = self.number_of_landhexes() / 7 + 0.02
        contingent = max([0, start_contingent - self.absolute_stock(color)])
        return contingent
    
    def mapfunction(self, hexf):
        # function for the dictionary map. returns '' if hexf out_of_bounds
        if self.out_of_bounds(hexf):
            return ''
        else:
            return self.map[hexf]
    
    def river_access(self, hexf):
        if self.map[hexf] in ['~~~', ' ~ '] or self.out_of_bounds(hexf):
            return '~'
        adj_river = False
        for direction in range(6):
            if self.mapfunction(self.next_hex(hexf,direction)) in ['~~~', ' ~ ']:
                if self.mapfunction(self.next_hex(hexf,(direction + 3) % 6)) in ['~~~', ' ~ ']:
                    return 2
                if self.mapfunction(self.next_hex(hexf,(direction + 2) % 6)) in ['~~~', ' ~ '] and not self.mapfunction(self.next_hex(hexf,(direction + 1) % 6)) in ['~~~', ' ~ ']:
                    return 2
                if self.mapfunction(self.next_hex(hexf,(direction - 2) % 6)) in ['~~~', ' ~ '] and not self.mapfunction(self.next_hex(hexf,(direction - 1) % 6)) in ['~~~', ' ~ ']:
                    return 2
                adj_river = True
        if adj_river:
            return 1
        else:
            return 0
        
    def river_access_score(self, color):
        score = 0
        for hexf in self.map.keys():
            if self.map[hexf] == color:
                score += self.river_access(hexf)
        return score
    
    def river_access_saldo(self,color):
        average_score = 0
        for col in self.colors[:7]:
            average_score += self.river_access_score(col)
        average_score = average_score / 7
        saldo = self.river_access_score(color) - average_score
        return saldo
                            
    def share_score(self, distance, color):
        #print(color,distance,self.contingent(color))
        share_score = round(distance**1.5 * self.contingent(color)**0.5 * max([1,6 - self.river_access_saldo(color)])**0.5 * max([0.4,4 - self.luck[color_index(color)]]) * 10)
        return share_score
    
    def create_map(self):
        #self.display_map()
        
        # randomize start field
        #start_y = random.randint(2,self.height - 3)
        #start_x = random.randint(2,self.width - ((start_y + self.form) % 2) - 3)
        #current_hex = (start_x,start_y)
        current_hex = self.start
        if self.map[current_hex] != ' ~ ':        
            dice = random.randint(0,6)
            self.map[current_hex] = self.colors[dice]
            luck = [-self.value_of_hex(current_hex) / 7]*7
            for color in range(len(luck)):
                if color == dice:
                    luck[color] += self.value_of_hex(current_hex)
                    break
            self.luck = np.array(luck, dtype=np.float64)
            #for col in range(len(luck)):
                #print(self.colors[col] + ': ' + str(luck[col]))
            #print()
            
        else:
            self.map[current_hex] = '~~~'
            #luck = np.array([0]*7, dtype=np.float64)
        #print(luck)
        #self.display_map()
        
        # randomize first step
        current_direction = random.randint(0,5)
        current_hex = self.next_hex(current_hex,current_direction)
        #print(current_direction, current_hex)
        #print(self.value_of_hex(current_hex))
        if self.out_of_bounds(current_hex):
            pass
        elif self.map[current_hex] == ' ~ ':
            self.map[current_hex] = '~~~'
        else:
            distances = {} 
            share_scores = []
            accumulated_scores = []
            for color in ['red','yel','bro','bla','blu','grn','gry']:
                distances[color] = self.work_distance(current_hex,color)
                score = round(abs(self.share_score(distances[color],color))**1)
                if len(accumulated_scores) == 0:
                    accumulated_scores += [score]
                else:
                    accumulated_scores += [accumulated_scores[-1] + score]
                share_scores += [score]
                #print(color, score)
            randint = random.randint(0,accumulated_scores[-1] - 1)
            for acsc in range(len(accumulated_scores)):
                if randint < accumulated_scores[acsc]:
                    dice = acsc
                    break
            self.map[current_hex] = self.colors[dice]
            luck_change = [-(distances[color]**1.5) * self.value_of_hex(current_hex) / sum([dis**1.5 for dis in distances.values()]) for color in ['red','yel','bro','bla','blu','grn','gry']]
            for color in range(len(luck_change)):
                if luck_change[color] == dice:
                    luck_change[color] += self.value_of_hex(current_hex)
                    #print(sum(luck_change))
                    break
            self.luck += np.array(luck_change, dtype=np.float64)
            #for col in range(len(luck)):
                #print(self.colors[col] + ': ' + str(luck[col]))
            #print()
        #self.display_map()
        current_direction = (current_direction + 2) % 6
        #print(current_hex, current_direction)
        
        # further steps
        while '???' in self.map.values() or ' ~ ' in self.map.values():
            #plot_map()
            while not self.out_of_bounds(self.next_hex(current_hex,current_direction)) and not self.map[self.next_hex(current_hex,current_direction)] in ['???',' ~ ']:
                #print('redirection, because current direction points at neither ??? nor  ~  nor out of map')
                current_direction = (current_direction - 1) % 6
                #print(current_hex, current_direction)
            current_hex = self.next_hex(current_hex,current_direction)
            #print(dir_str(current_direction) + ' to ' +str(current_hex))
            #print(self.value_of_hex(current_hex))
            if self.out_of_bounds(current_hex):
                pass
            elif self.map[current_hex] == ' ~ ':
                self.map[current_hex] = '~~~'
            else:
                distances = {} 
                share_scores = []
                accumulated_scores = []
                for color in ['red','yel','bro','bla','blu','grn','gry']:
                    distances[color] = self.work_distance(current_hex,color)
                    score = round(abs(self.share_score(distances[color],color))**1)
                    if len(accumulated_scores) == 0:
                        accumulated_scores += [score]
                    else:
                        accumulated_scores += [accumulated_scores[-1] + score]
                    share_scores += [score]
                    #print(color, score)
                randint = random.randint(0,accumulated_scores[-1] - 1)
                for acsc in range(len(accumulated_scores)):
                    if randint < accumulated_scores[acsc]:
                        dice = acsc
                        break
                self.map[current_hex] = self.colors[dice]
                luck_change = [-(distances[color]**1.5) * self.value_of_hex(current_hex) / sum([dis**1.5 for dis in distances.values()]) for color in ['red','yel','bro','bla','blu','grn','gry']]
                #print(dice)
                for color in range(len(luck_change)):
                    if color == dice:
                        #print(self.colors[color] + ' won')
                        luck_change[color] += self.value_of_hex(current_hex)
                        #print(sum(luck_change))
                        break
                self.luck += np.array(luck_change, dtype=np.float64)
                #for col in range(len(luck)):
                    #print(self.colors[col] + ': ' + str(luck[col]))
                #print()
                #print(sum(luck))
                #self.display_map()
            #for color in self.colors[:7]:
                #print(color + ': ' + str(self.contingent(color)))
            current_direction = (current_direction + 1) % 6
        
        #print('Contingents:')
        #for color in self.colors[:7]:
        #    print(color + ': ' + str(round(self.contingent(color),2)))
        #print('River access:')
        #for color in self.colors[:7]:
        #    print(color + ': ' + str(self.river_access_score(color)))
        #print('Luck:')
        luckstr = 'Luck: '
        for col in range(len(self.luck)):
            luckstr += self.colors[col] + ': ' + str(round(self.luck[col],2)) + '; '
        luckstr = luckstr + '\n' + 'River access: '
        for color in self.colors[:7]:
            luckstr = luckstr + color + ': ' + str(self.river_access_score(color)) + '; '
        luckstr = luckstr + '\n'
        nolh = self.number_of_landhexes()
        for color in self.colors[:7]:
            delta = self.absolute_stock(color) - (nolh / 7)
            if delta >= 1:
                luckstr = luckstr + color + ' has +' + str(round(delta,1)) + ' hex;'
            if delta <= -1:
                luckstr = luckstr + color + ' has ' + str(round(delta,1)) + ' hex; '
        return luckstr
        
    def number_test(self):
        nolh = self.number_of_landhexes()
        if nolh % 7 == 0:
            av = nolh // 7
            totaldelta = 0
            for color in self.colors[:7]:
                delta = abs(self.absolute_stock(color) - (nolh / 7))
                #print(color,delta)
                totaldelta += delta
            if totaldelta > 3:
                #print(totaldelta)
                return False
            else:
                return True
        else:
            for color in self.colors[:7]:
                delta = abs(self.absolute_stock(color) - (nolh / 7))
                if delta > 1:
                    #print(color,delta)
                    return False
            return True
        
    def river_access_test(self):
        for color in self.colors[:7]:
            saldo = self.river_access_saldo(color)
            if saldo > 2:
                #print(color,saldo)
                return False
        return True
    
    def luck_test(self):
        for color in range(7):
            luck = self.luck[color]
            if luck > 1.5:
                #print(self.colors[color],luck)
                return False
        return True
        
        
    def display_map(self):
        for row in range(self.height):
            if (row + self.form) % 2 == 1:
                rowstr = '  '
            else:
                rowstr = ''
            for col in range(self.width - ((row + self.form) % 2)):
                rowstr += self.map[(col,row)] + ' '
            print(rowstr)
        print('\n')
        
    def display_final_map_colored(self):
        for row in range(self.height):
            if (row + self.form) % 2 == 1:
                rowstr = '---'
            else:
                rowstr = ''
            for col in range(self.width - ((row + self.form) % 2)):
                rowstr += transform_string_to_box(self.map[(col,row)])
            display(HTML(rowstr))
            
    def bga_format(self):
        bga_format = ''
        rows = []
        for row in range(self.height):
            rows += [','.join([bga_symbol(self.map[(col,row)]) for col in range(self.width - ((row + self.form) % 2))])]
        for row in rows:
            print(row)
        bga_format = '\n'.join(rows)
        return bga_format
    
    def river_access_map(self):
        bga_format = ''
        rows = []
        for row in range(self.height):
            if (row + self.form) % 2 == 0:
                rowstr = ''
            else:
                rowstr = ' '
            rowstr = rowstr + ','.join([str(self.river_access((col,row))) for col in range(self.width - ((row + self.form) % 2))])
            rows += [rowstr]
        for row in rows:
            print(row)
        bga_format = '\n'.join(rows)
        #return bga_format

    def next_hex(self,hexf,direction):
        if direction == 0:
            if (hexf[1] + self.form) % 2 == 0:
                return (hexf[0],hexf[1] + 1)
            else:
                return (hexf[0] + 1,hexf[1] + 1)
        if direction == 1:
            return (hexf[0] + 1,hexf[1])
        if direction == 2:
            if (hexf[1] + self.form) % 2 == 0:
                return (hexf[0],hexf[1] - 1)
            else:
                return (hexf[0] + 1,hexf[1] - 1)
        if direction == 3:
            if (hexf[1] + self.form) % 2 == 0:
                return (hexf[0] - 1,hexf[1] - 1)
            else:
                return (hexf[0],hexf[1] - 1)
        if direction == 4:
            return (hexf[0] - 1,hexf[1])
        if direction == 5:
            if (hexf[1] + self.form) % 2 == 0:
                return (hexf[0] - 1,hexf[1] + 1)
            else:
                return (hexf[0],hexf[1] + 1)
    def painter(self,ax):
        plot_colors ={
                    'red': 'red',
                    'yel': 'yellow',
                    'bro': '#835C3B',
                    'bla': 'black',
                    'blu': 'blue',
                    'grn': 'green',
                    'gry': '#808080',
                    '~~~': '#BFEFFF',
                    ' ~ ': '#BFEFFF',
                    '???': 'orange'
                    }
        hexes = self.map
        for key in hexes:
            hex_x = key[1]
            hex_y = key[0]
            align_hex = 0.5 if (hex_x + self.form) % 2 ==1 else 1
            alphaval = 0.2 if hexes[key] == '~~~' else 0.7
            hex_col = plot_colors[hexes[key]]
            hexagon = RegularPolygon((hex_y - align_hex, -hex_x), numVertices=6, radius=np.sqrt(1 / 3),
                                    alpha=alphaval, fill=True, color=hex_col)
            ax.add_patch(hexagon)
    
def path_cost(transit_colors):
    destination_color = transit_colors[-1]
    if ['~~~','~~~','~~~','~~~'] in [transit_colors[i:i + 4] for i in range(len(transit_colors) - 3)]:
        if destination_color == 'blu':
            path_work = 42 * 6
        #elif destination_color == 'yel':
        #    path_work = 378 - 4 * 21
        elif destination_color == 'gry':
            path_work = 378 - 4 * 42
        else:
            path_work = 42 * 7
    elif ['~~~','~~~','~~~'] in [transit_colors[i:i + 3] for i in range(len(transit_colors) - 2)]:
        if destination_color == 'blu':
            path_work = 42 * 4
        #elif destination_color == 'yel':
        #    path_work = 42 * 35/6 - 3 * 21
        elif destination_color == 'gry':
            path_work = 42 * 35/6 - 3 * 42
        else:
            path_work = 42 * 5
    elif ['~~~','~~~'] in [transit_colors[i:i + 2] for i in range(len(transit_colors) - 1)]:
        if destination_color == 'blu':
            path_work = 42 * 10/6
        #elif destination_color == 'yel':
        #    path_work = 42 * 22/6 - 2 * 21
        elif destination_color == 'gry':
            path_work = 42 * 22/6 - 2 * 42
        else:
            path_work = 42 * 3
    elif ['~~~'] in [transit_colors[i:i + 1] for i in range(len(transit_colors))]:
        if destination_color == 'blu':
            path_work = 42 * 2/6
        #elif destination_color == 'yel':
        #    path_work = 42 * 7/6 - 1 * 21
        elif destination_color == 'gry':
            path_work = 42 * 9/6 - 1 * 42
        else:
            path_work = 42
    else:
        path_work = 0
    for tcn in range(len(transit_colors)):
        current_color = transit_colors[tcn]
        trans_work = 42*color_distance(destination_color,transit_colors[tcn])
        if destination_color == 'red': 
            if current_color not in ['red','~~~',' ~ ']:
                trans_work = (trans_work + 42 * 8/6)/2
        if destination_color == 'yel': 
            if current_color not in ['yel','red','bro','~~~',' ~ '] and (tcn == 0 or transit_colors[tcn - 1] not in ['~~~',' ~ '] or transit_colors[tcn + 1] not in ['~~~',' ~ ']):
                trans_work = min([trans_work, 42 * 8/6])
        if destination_color == 'bla':
            trans_work = trans_work * 5/6
        if destination_color == 'gry':
            if len(transit_colors) == 2:
                if transit_colors[0] in ['~~~',' ~ ']:
                    trans_work = trans_work * 10/6
                else:
                    trans_work = trans_work * 9/6
            else:
                trans_work = trans_work * 4/6
        if destination_color == 'grn':
            trans_work = trans_work * 8/6
        path_work += trans_work
    return path_work

def dir_str(number):
    if number == 0:
        return 'down-right'
    elif number == 1:
        return 'right'
    elif number == 2:
        return 'up-right'
    elif number == 3:
        return 'up-left'
    elif number == 4:
        return 'left'
    elif number == 5:
        return 'down-left'

def color_index(color):
    if color == 'red':
        return 0
    elif color == 'yel':
        return 1
    elif color == 'bro':
        return 2
    elif color == 'bla':
        return 3
    elif color == 'blu':
        return 4
    elif color == 'grn':
        return 5
    elif color == 'gry':
        return 6
    else:
        return 100

def color_distance(color1, color2):
    if 'gry' in [color1,color2] and '~~~' in [color1,color2]:
        distance = 1
    elif 'yel' in [color1,color2] and '~~~' in [color1,color2]:
        distance = 0.5
    elif color1 in [' ~ ','~~~'] or color2 in [' ~ ','~~~']:
        distance = 0
    elif color1 == '???' or color2 == '???':
        distance = 1.5
    else:
        distance = min([(color_index(color1) - color_index(color2)) % 7,(color_index(color2) - color_index(color1)) % 7])
    return distance
   
def transform_string_to_box(color_string):
    color_mapping = {
        'red': (' ', 'red'),
        'yel': (' ', 'yellow'),
        'bro': (' ', '#835C3B'),
        'bla': (' ', 'black'),
        'blu': (' ', 'blue'),
        'grn': (' ', 'green'),
        'gry': (' ', '#808080'),
        '~~~': (' ', '#BFEFFF')
    }

    if color_string in color_mapping:
        display_text, color = color_mapping[color_string]
        box_html = f'<div style="background-color: {color}; width: 30px; height: 25px; text-align: center; display: inline-block; margin-bottom: -20px;">{display_text[:4]}</div>'
        return box_html
    else:
        return color_string 

def bga_symbol(color):
    if color == 'red':
        return 'R'
    elif color == 'yel':
        return 'Y'
    elif color == 'bro':
        return 'U'
    elif color == 'bla':
        return 'K'
    elif color == 'blu':
        return 'B'
    elif color == 'grn':
        return 'G'
    elif color == 'gry':
        return 'S'
    elif color == '~~~':
        return 'I'
                
