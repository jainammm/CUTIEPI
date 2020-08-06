import numpy as np
from collections import defaultdict
from models.tokenization import FullTokenizer
import random

import unicodedata
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass 
    return False

class DataLoader:
    def __init__(self, params, json_data, update_dict=True, load_dictionary=False):

        self.special_dict = {'*', '='} # map texts to specific tokens
        self.encoding_factor = 1 # ensures the size (rows/cols) of grid table compat with the network 
        self.segment_grid = params.segment_grid if hasattr(params, 'segment_grid') else False # segment grid into two parts if grid is larger than cols_target
        self.cols_segment = params.cols_segment if hasattr(params, 'cols_segment') else 72
        self.da_extra_rows = params.data_augmentation_extra_rows if hasattr(params, 'data_augmentation_extra_rows') else 0 # randomly expand rows/cols
        self.da_extra_cols = params.data_augmentation_extra_cols if hasattr(params, 'data_augmentation_extra_cols') else 0 # randomly expand rows/cols
        self.augment_strategy = params.augment_strategy if hasattr(params, 'augment_strategy') else 1
        self.rows_target = params.rows_target if hasattr(params, 'rows_target') else 80 
        self.cols_target = params.cols_target if hasattr(params, 'cols_target') else 80 
        self.rows_ulimit = params.rows_ulimit if hasattr(params, 'rows_ulimit') else 80 # handle OOM, must be multiple of self.encoding_factor
        self.cols_ulimit = params.cols_ulimit if hasattr(params, 'cols_ulimit') else 80 # handle OOM, must be multiple of self.encoding_factor
        
        self.pm_strategy = params.positional_mapping_strategy if hasattr(params, 'positional_mapping_strategy') else 2 
        self.data_mode = 2 # 0 to consider key and value as two different class, 1 the same class, 2 only value considered
        self.text_case = params.text_case 
        self.tokenize = params.tokenize
        if self.tokenize:
            self.tokenizer = FullTokenizer('dict/vocab.txt', do_lower_case=not self.text_case)

        ## 0> parameters to be tuned
        self.load_dictionary = load_dictionary # load dictionary from file rather than start from empty 
        self.dict_path = params.load_dict_from_path if load_dictionary else params.dict_path
        if self.load_dictionary:
            self.dictionary = np.load(self.dict_path + '_dictionary.npy').item()
            self.word_to_index = np.load(self.dict_path + '_word_to_index.npy').item()
            self.index_to_word = np.load(self.dict_path + '_index_to_word.npy').item()
        else:
            self.dictionary = {'[PAD]':0, '[UNK]':0} # word/counts. to be updated in self.load_data() and self._update_docs_dictionary()
            self.word_to_index = {}
            self.index_to_word = {}

        self.num_words = len(self.dictionary)   

        self.classes = ['DontCare','SELLER_STATE', 'SELLER_ID', 'SELLER_NAME', 'SELLER_ADDRESS', 'SELLER_GSTIN_NUMBER',
            'COUNTRY_OF_ORIGIN', 'CURRENCY', 'DESCRIPTION', 'INVOICE_NUMBER', 'INVOICE_DATE', 'DUE_DATE',
            'TOTAL_INVOICE_AMOUNT_ENTERED_BY_WH_OPERATOR', 'PO_NUMBER', 'BUYER_GSTIN_NUMBER', 'SHIP_TO_ADDRESS',
            'PRODUCT_ID', 'HSN', 'TITLE', 'QUANTITY', 'UNIT_PRICE', 'DISCOUNT_PERCENT', 'SGST_PERCENT',
            'CGST_PERCENT', 'IGST_PERCENT', 'TOTAL_AMOUNT']

        # self.classes = ['DontCare', 'SELLER_NAME', 'SELLER_GSTIN_NUMBER', 
        #     'TOTAL_INVOICE_AMOUNT_ENTERED_BY_WH_OPERATOR', 'HSN', 'TITLE']

        self.num_classes = len(self.classes) 

        self.validation_docs =  self.load_data(json_data, update_dict=update_dict)


    def load_data(self, data, update_dict=False):
        """
        label_dressed in format:
        {file_id: {class: [{'key_id':[], 'value_id':[], 'key_text':'', 'value_text':''}, ] } }
        load doc words with location and class returned in format: 
        [[file_name, text, word_id, [x_left, y_top, x_right, y_bottom], [left, top, right, bottom], max_row_words, max_col_words] ]
        """
        
        doc_dressed = []
        file_id = data['global_attributes']['file_id']
        
        data = self._collect_data(file_id, data['text_boxes'], update_dict)
        for i in data:
            doc_dressed.append(i)
                    
        return doc_dressed 

    def fetch_validation_data(self):
        
        validation_docs = self.validation_docs
        
        ## fixed validation shape leads to better result (to be verified)
        real_rows, real_cols, _, _ = self._cal_rows_cols(validation_docs, extra_augmentation=False)
        rows = max(self.rows_target, real_rows)
        cols = max(self.rows_target, real_cols)
        
        grid_table, gt_classes, bboxes, label_mapids, bbox_mapids, file_names, updated_cols, _, _, text_ids = \
            self._positional_mapping(validation_docs, rows, cols)   
        if updated_cols > cols:
            print('Validation grid EXPAND size: ({},{}) from ({},{})'\
                    .format(rows, updated_cols, rows, cols))
            grid_table, gt_classes, bboxes, label_mapids, bbox_mapids, file_names, _, _, _, text_ids = \
                self._positional_mapping(validation_docs, rows, updated_cols)     
        
        ## load image and generate corresponding @ps_1dindices
        images, ps_1d_indices = [], []
        
        batch = {'grid_table': np.array(grid_table), 'gt_classes': np.array(gt_classes), 
                 'data_image': np.array(images), 'ps_1d_indices': np.array(ps_1d_indices), # @images and @ps_1d_indices are only used for CUTIEv2
                 'bboxes': bboxes, 'label_mapids': label_mapids, 'bbox_mapids': bbox_mapids,
                 'file_name': file_names, 'shape': [rows,cols], 'text_ids' : text_ids}
        return batch
    

    def _collect_data(self, file_name, content, update_dict):
        """
        dress and preserve only interested data.
        """          
        content_dressed = []
        left, top, right, bottom, buffer = 9999, 9999, 0, 0, 2
        for line in content:
            bbox = line['bbox'] # handle data corrupt
            if len(bbox) == 0:
                continue
            if line['text'] in self.special_dict: # ignore potential overlap causing characters
                continue
            
            x_left, y_top, x_right, y_bottom = self._dress_bbox(bbox)        
            # TBD: the real image size is better for calculating the relative x/y/w/h
            if x_left < left: left = x_left - buffer
            if y_top < top: top = y_top - buffer
            if x_right > right: right = x_right + buffer
            if y_bottom > bottom: bottom = y_bottom + buffer
            
            word_id = line['id']
            dressed_texts = self._dress_text(line['text'], update_dict)
            
            num_block = len(dressed_texts)
            for i, dressed_text in enumerate(dressed_texts): # handling tokenized text, separate bbox
                new_left = int(x_left + (x_right-x_left) / num_block * (i))
                new_right = int(x_left + (x_right-x_left) / num_block * (i+1))
                content_dressed.append([file_name, dressed_text, word_id, [new_left, y_top, new_right, y_bottom]])
            
        # initial calculation of maximum number of words in rows/cols in terms of image size
        num_words_row = [0 for _ in range(bottom)] # number of words in each row
        num_words_col = [0 for _ in range(right)] # number of words in each column
        for line in content_dressed:
            _, _, _, [x_left, y_top, x_right, y_bottom] = line
            for y in range(y_top, y_bottom):
                num_words_row[y] += 1
            for x in range(x_left, x_right):
                num_words_col[x] += 1
        max_row_words = self._fit_shape(max(num_words_row))
        max_col_words = 0#self._fit_shape(max(num_words_col))
        
        # further expansion of maximum number of words in rows/cols in terms of grid shape
        max_rows = max(self.encoding_factor, max_row_words)
        max_cols = max(self.encoding_factor, max_col_words)
        DONE = False
        while not DONE:
            DONE = True
            grid_table = np.zeros([max_rows, max_cols], dtype=np.int32)
            for line in content_dressed:
                _, _, _, [x_left, y_top, x_right, y_bottom] = line
                row = int(max_rows * (y_top - top + (y_bottom-y_top)/2) / (bottom-top))
                col = int(max_cols * (x_left - left + (x_right-x_left)/2) / (right-left))
                
                while col < max_cols and grid_table[row, col] != 0: # shift to find slot to drop the current item
                    col += 1
                if col == max_cols: # shift to find slot to drop the current item
                    col -= 1
                    ptr = 0
                    while ptr<max_cols and grid_table[row, ptr] != 0:
                        ptr += 1
                    if ptr == max_cols: # overlap cannot be solved in current row, then expand the grid
                        max_cols = self._expand_shape(max_cols)
                        DONE = False
                        break
                    
                    grid_table[row, ptr:-1] = grid_table[row, ptr+1:]
                
                if DONE:
                    if row > max_rows or col>max_cols:
                        print('wrong')
                    grid_table[row, col] = 1
        
        max_rows = self._fit_shape(max_rows)
        max_cols = self._fit_shape(max_cols)
        
        #print('{} collected in shape: {},{}'.format(file_name, max_rows, max_cols))
        
        # segment grid into two parts if number of cols is larger than self.cols_target
        data = []
        if self.segment_grid and max_cols > self.cols_segment:
            content_dressed_left = []
            content_dressed_right = []
            cnt = defaultdict(int) # counter for number of words in a specific row
            cnt_l, cnt_r = defaultdict(int), defaultdict(int) # update max_cols if larger than self.cols_segment
            left_boundary = max_cols - self.cols_segment
            right_boundary = self.cols_segment
            for i, line in enumerate(content_dressed):
                file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom] = line
                
                row = int(max_rows * (y_top + (y_bottom-y_top)/2) / bottom)
                cnt[row] += 1                
                if cnt[row] <= left_boundary:
                    cnt_l[row] += 1
                    content_dressed_left.append([file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                                      [left, top, right, bottom], max_rows, self.cols_segment])
                elif left_boundary < cnt[row] <= right_boundary:
                    cnt_l[row] += 1
                    cnt_r[row] += 1
                    content_dressed_left.append([file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                                      [left, top, right, bottom], max_rows, self.cols_segment])
                    content_dressed_right.append([file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                                      [left, top, right, bottom], max_rows, max(max(cnt_r.values()), self.cols_segment)])
                else:
                    cnt_r[row] += 1
                    content_dressed_right.append([file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                                      [left, top, right, bottom], max_rows, max(max(cnt_r.values()), self.cols_segment)])
    
            if max(cnt_l.values()) < 2*self.cols_segment:
                data.append(content_dressed_left)
            if max(cnt_r.values()) < 2*self.cols_segment: # avoid OOM, which tends to happen in the right side
                data.append(content_dressed_right)
        else:
            for i, line in enumerate(content_dressed): # append height/width/numofwords to the list
                file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom] = line
                content_dressed[i] = [file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                                      [left, top, right, bottom], max_rows, max_cols ]
            data.append(content_dressed)
        return data

    def _dress_bbox(self, bbox):
        positions = np.array(bbox).reshape([-1])
        x_left = max(0, min(positions[0::2]))
        x_right = max(positions[0::2])
        y_top = max(0, min(positions[1::2]))
        y_bottom = max(positions[1::2])
        return int(x_left), int(y_top), int(x_right), int(y_bottom)   

    def _dress_text(self, text, update_dict):
        """
        three cases covered: 
        alphabetic string, numeric string, special character
        """
        string = text if self.text_case else text.lower()
        for i, c in enumerate(string):
            if is_number(c):
                string = string[:i] + '0' + string[i+1:]
                
        strings = [string]
        if self.tokenize:
            strings = self.tokenizer.tokenize(strings[0])
            #print(string, '-->', strings)
            
        for idx, string in enumerate(strings):            
            if string.isalpha():
                if string in self.special_dict:
                    string = self.special_dict[string]
                # TBD: convert a word to its most similar word in a known vocabulary
            elif is_number(string):
                pass
            elif len(string)==1: # special character
                pass
            else:
                # TBD: seperate string as parts for alpha and number combinated strings
                #string = re.findall('[a-z]+', string)
                pass            
            
            if string not in self.dictionary.keys():
                if update_dict:
                    self.dictionary[string] = 0
                else:
                    #print('unknown text: ' + string)
                    string = '[UNK]' # TBD: take special care to unmet words\
            self.dictionary[string] += 1
            
            strings[idx] = string
        return strings
    
    def _fit_shape(self, shape): # modify shape size to fit the encoding factor
        while shape % self.encoding_factor:
            shape += 1
        return shape

    def _expand_shape(self, shape): # expand shape size with step 2
        return self._fit_shape(shape+1)

    def _cal_rows_cols(self, docs, extra_augmentation=False, dropout=False):                  
        max_row = self.encoding_factor
        max_col = self.encoding_factor
        for doc in docs:
            for line in doc: 
                _, _, _, _, _, max_row_words, max_col_words = line
                if max_row_words > max_row:
                    max_row = max_row_words
                if max_col_words > max_col:
                    max_col = max_col_words
        
        pre_rows = self._fit_shape(max_row) #(max_row//self.encoding_factor+1) * self.encoding_factor
        pre_cols = self._fit_shape(max_col) #(max_col//self.encoding_factor+1) * self.encoding_factor
        
        rows, cols = 0, 0
        if extra_augmentation:
            pad_row = int(random.gauss(0, self.da_extra_rows*self.encoding_factor)) #abs(random.gauss(0, u))
            pad_col = int(random.gauss(0, self.da_extra_cols*self.encoding_factor)) #random.randint(0, u)
            
            if self.augment_strategy == 1: # strategy 1: augment data by increasing grid shape sizes
                pad_row = abs(pad_row)
                pad_col = abs(pad_col)
                rows = self._fit_shape(max_row+pad_row) # apply upper boundary to avoid OOM
                cols = self._fit_shape(max_col+pad_col) # apply upper boundary to avoid OOM
            elif self.augment_strategy == 2 or self.augment_strategy == 3: # strategy 2: augment by increasing or decreasing the target gird shape size
                rows = self._fit_shape(max(self.rows_target+pad_row, max_row)) # protect grid shape
                cols = self._fit_shape(max(self.cols_target+pad_col, max_col)) # protect grid shape
            else:
                raise Exception('unknown augment strategy')
            rows = min(rows, self.rows_ulimit) # apply upper boundary to avoid OOM
            cols = min(cols, self.cols_ulimit) # apply upper boundary to avoid OOM                                
        else:
            rows = pre_rows
            cols = pre_cols
        return rows, cols, pre_rows, pre_cols 

    def _dress_class(self, file_name, word_id, labels):
        """
        label_dressed in format:
        {file_id: {class: [{'key_id':[], 'value_id':[], 'key_text':'', 'value_text':''}, ] } }
        """
        if file_name in labels:
            for cls, cls_labels in labels[file_name].items():
                for idx, cls_label in enumerate(cls_labels):
                    for key, values in cls_label.items():
                        if (key=='key_id' or key=='value_id') and word_id in values:
                            if key == 'key_id':
                                if self.data_mode == 0:
                                    return idx, self.classes.index(cls) * 2 - 1 # odd
                                elif self.data_mode == 1:
                                    return idx, self.classes.index(cls)
                                else: # ignore key_id when self.data_mode is not 0 or 1
                                    return 0, 0
                            elif key == 'value_id':
                                if self.data_mode == 0:
                                    return idx, self.classes.index(cls) * 2 # even 
                                else: # when self.data_mode is 1 or 2
                                    return idx, self.classes.index(cls) 
            return 0, 0 # 0 is of class type 'DontCare'
        print("No matched labels found for {}".format(file_name))
    
    
    def _positional_mapping(self, docs, rows, cols):
        """
        docs in format:
        [[file_name, text, word_id, [x_left, y_top, x_right, y_bottom], [left, top, right, bottom], max_row_words, max_col_words] ]
        return grid_tables, gird_labels, dict bboxes {file_name:[]}, file_names
        """
        grid_tables = []
        gird_labels = []
        ps_indices_x = [] # positional sampling indices
        ps_indices_y = [] # positional sampling indices
        bboxes = {}
        text_ids = {}
        label_mapids = []
        bbox_mapids = [] # [{}, ] bbox identifier, each id with one or multiple bbox/bboxes
        file_names = []
        for doc in docs:
            items = []
            cols_e = 2 * cols # use @cols_e larger than required @cols as buffer
            grid_table = np.zeros([rows, cols_e], dtype=np.int32)
            grid_label = np.zeros([rows, cols_e], dtype=np.int8)
            ps_x = np.zeros([rows, cols_e], dtype=np.int32)
            ps_y = np.zeros([rows, cols_e], dtype=np.int32)
            bbox = [[] for c in range(cols_e) for r in range(rows)]
            text_id_ = [[] for c in range(cols_e) for r in range(rows)]
            bbox_id, bbox_mapid = 0, {} # one word in one or many positions in a bbox is mapped in bbox_mapid
            label_mapid = [[] for _ in range(self.num_classes)] # each class is connected to several bboxes (words)

            for item in doc:
                file_name = item[0]
                text = item[1]
                text_id = item[2]
                x_left, y_top, x_right, y_bottom = item[3][:]
                left, top, right, bottom = item[4][:]
                
                dict_id = self.word_to_index[text]                
                # entity_id, class_id = self._dress_class(file_name, word_id, labels)
                
                bbox_id += 1 

                box_y = y_top + (y_bottom-y_top)/2
                box_x = x_left # h_l is used for image feature map positional sampling
                v_c = (y_top - top + (y_bottom-y_top)/2) / (bottom-top)
                h_c = (x_left - left + (x_right-x_left)/2) / (right-left) # h_c is used for sorting items
                row = int(rows * v_c) 
                col = int(cols * h_c) 
                items.append([row, col, [box_y, box_x], [v_c, h_c], file_name, dict_id, bbox_id, [x_left, y_top, x_right-x_left, y_bottom-y_top], text_id])                       
            
            items.sort(key=lambda x: (x[0], x[3], x[5])) # sort according to row > h_c > bbox_id
            for item in items:
                row, col, [box_y, box_x], [v_c, h_c], file_name, dict_id, bbox_id, box, text_id = item
                
                while col < cols and grid_table[row, col] != 0:
                    col += 1            
                
                # self.pm_strategy 0: skip if overlap
                # self.pm_strategy 1: shift to find slot if overlap
                # self.pm_strategy 2: expand grid table if overlap
                if self.pm_strategy == 0:
                    if col == cols:                     
                        print('overlap in {} row {} r{}c{}!'.
                              format(file_name, row, rows, cols))
                        #print(grid_table[row,:])
                        #print('overlap in {} <{}> row {} r{}c{}!'.
                        #      format(file_name, self.index_to_word[dict_id], row, rows, cols))
                    else:
                        grid_table[row, col] = dict_id                     
                        bbox_mapid[row*cols+col] = bbox_id                       
                        bbox[row*cols+col] = box
                        text_id_[row*cols+col] = text_id   
                elif self.pm_strategy==1 or self.pm_strategy==2:
                    ptr = 0
                    if col == cols: # shift to find slot to drop the current item
                        col -= 1
                        while ptr<cols and grid_table[row, ptr] != 0:
                            ptr += 1
                        if ptr == cols:
                            grid_table[row, :-1] = grid_table[row, 1:]
                        else:
                            grid_table[row, ptr:-1] = grid_table[row, ptr+1:]
                        
                    if self.pm_strategy == 2:
                        while col < cols_e and grid_table[row, col] != 0:
                            col += 1
                        if col > cols: # update maximum cols in current grid
                            print(grid_table[row,:col])
                            print('overlap in {} <{}> row {} r{}c{}!'.
                                  format(file_name, self.index_to_word[dict_id], row, rows, cols))
                            cols = col
                        if col == cols_e:      
                            print('overlap!')
                    
                    grid_table[row, col] = dict_id
                    ps_x[row, col] = box_x
                    ps_y[row, col] = box_y
                    bbox_mapid[row*cols+col] = bbox_id     
                    bbox[row*cols+col] = box
                    text_id_[row*cols+col] = text_id
                
            cols = self._fit_shape(cols)
            grid_table = grid_table[..., :cols]
            grid_label = grid_label[..., :cols]
            ps_x = np.array(ps_x[..., :cols])
            ps_y = np.array(ps_y[..., :cols])
            
            grid_tables.append(np.expand_dims(grid_table, -1)) 
            gird_labels.append(grid_label) 
            ps_indices_x.append(ps_x)
            ps_indices_y.append(ps_y)
            bboxes[file_name] = bbox
            text_ids[file_name] = text_id_
            label_mapids.append(label_mapid)
            bbox_mapids.append(bbox_mapid)
            file_names.append(file_name)
            
        return grid_tables, gird_labels, bboxes, label_mapids, bbox_mapids, file_names, cols, ps_indices_x, ps_indices_y, text_ids
    