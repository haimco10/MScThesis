f = figure;
set(f,'Position',[500 500 300 150]);
dat =  {'Task 1', ' Q', 'NQ',' NQ','NQ','NQ';
        'Task 2', ' NQ', 'NQ',' NQ','Q','NQ';
        'Task 3', ' NQ', 'NQ',' NQ','NQ','Q';
        'Task 4', ' NQ', 'Q',' Q','NQ','NQ';
};
columnname =   {'time: ', '1', '2','3','4','5'};
columnformat = {'char', 'char', 'char','char','char','char';}; 
t = uitable('Units','normalized','Position',...
            [0.05 0.05 0.755 0.87], 'Data', dat,... 
            'ColumnName', columnname,...
            'ColumnFormat', columnformat,...
            'RowName',[]); 
       