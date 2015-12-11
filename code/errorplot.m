function errorplot(e,n,c)
 plot(cumsum( (1-n).*e + (1-e).*n), c);
