select * from product_type

select * from product

select pt.name product_type, p.name product, pt.product_type_cd 
from product p
inner join product_type pt on pt.product_type_cd = p.product_type_cd 
where pt.name = 'Customer Accounts'

select pt.name product_type, p.name product, pt.product_type_cd 
from product p
inner join product_type pt on pt.product_type_cd = p.product_type_cd 
where pt.name != 'Customer Accounts'


select * from employee where start_date > '2001-01-01'

select * from employee where date_part('year',start_date) > '2002'

select * from employee where date_part('year', start_date) between '2003' and '2007'

select * from account where avail_balance between 3000 and 5000;


select * from customer c 
where c.cust_type_cd = 'I'

select * from customer c 
where c.cust_type_cd = 'I' and fed_id between '500-00-0000' and '999-99-9999'

select * from account where product_cd in ('CHK', 'SAV', 'CD', 'MM')

select product_cd , * from product

select * from account where product_cd in (select product_cd from product where product_type_cd = 'ACCOUNT')
-- same as above
select * from account a
inner join product p on p.product_cd = a.product_cd 
where p.product_type_cd = 'ACCOUNT'

select * from employee where superior_emp_id != 6 or superior_emp_id is null 


select * from account
-- all accounts opened in 2002
select * from account 
where date_part('year', open_date) = '2002'
