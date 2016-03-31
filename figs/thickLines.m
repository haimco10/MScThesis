function thickLines(n)
% thickLines  Make plot lines thicker and use larger fonts than MATLAB default.  Use this
%             utility to create plots that are legible after resizing in LaTeX or with
%             a word processor.
%
% Synopsis:  thickLines
%            thickLines(n)
%
% Input:  n = (optional, integer) degree of thickening.  n = 1,2,3 for increasing
%              degree of line thickness and default font size.  Default:  n=2
%
% Output:  None.  The result is that the line thickness is changed for all *new*
%          plots created in the current MATLAB session.

% Trick from L.N. Trefethen, "Spectral Methods in MATLAB", 2000, SIAM

if nargin<1,  n = 2;  end

switch n
    case 0, % return to Matlab's default
        set(0,'defaultaxesfontsize',10,'defaultaxeslinewidth',0.5,...
            'defaultlinelinewidth',0.5,'defaultpatchlinewidth',0.5);
    case 1,
        set(0,'defaultaxesfontsize',12,'defaultaxeslinewidth',0.7,...
            'defaultlinelinewidth',1,'defaultpatchlinewidth',0.7);
    case 2,
        set(0,'defaultaxesfontsize',14,'defaultaxeslinewidth',0.9,...
            'defaultlinelinewidth',2,'defaultpatchlinewidth',0.9);
    case 3,
        set(0,'defaultaxesfontsize',16,'defaultaxeslinewidth',1,...
            'defaultlinelinewidth',2.5,'defaultpatchlinewidth',1);
    case 4,
        set(0,'defaultaxesfontsize',18,'defaultaxeslinewidth',1.3,...
            'defaultlinelinewidth',3,'defaultpatchlinewidth',1.2);
    otherwise,
        error(sprintf('n = %d not allowed'));
        
end