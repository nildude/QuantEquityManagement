-- get the employee associated with each department
select * from employee e 
select * from department d
select d.name, e.fname, e.lname from employee e
inner join department d on d.dept_id = e.dept_id 

-- return all accounts opened by tellers hired prior to 2007 currently assigned to wobourn branch
select * from branch
select * from account
select * from employee

select a.account_id, a.cust_id, a.open_date, a.product_cd from account a
inner join branch b on b.branch_id = a.open_branch_id 
inner join employee e on e.emp_id = a.open_emp_id 
where date_part('year',e.start_date) < '2007' 
and (e.title = 'Teller' or e.title = 'Head Teller')
and b.name = 'Woburn Branch'

-- return the account id and federal tax number for all business accounts
select a.account_id, c.fed_id from account a 
inner join customer c on c.cust_id = a.cust_id 
where c.cust_type_cd = 'B'

-- get the name of the teller who opened each account
select a.account_id, c.fed_id, e.fname, e.lname, e.title from account a 
inner join customer c on c.cust_id = a.cust_id 
inner join employee e on e.emp_id = a.open_emp_id 
where c.cust_type_cd = 'B'

--using subqueries
-- this method is the fastest
--return all accounts opened by tellers hired prior to 2007 currently assigned to woburn branch
select a.account_id, a.cust_id, a.open_date, a.product_cd from account a
where a.open_branch_id in (select b.branch_id from branch b where b.name = 'Woburn Branch')
and a.open_emp_id in (select e.emp_id from employee e 
where e.title = 'Teller' or e.title = 'Head Teller' and date_part('year',e.start_date) < '2007')

-- another option
select a.account_id, a.cust_id, a.open_date, a.product_cd from account a
inner join (select emp_id from employee
where date_part('year', start_date) < '2007' 
and (title = 'Teller' or title = 'Head Teller')) e on e.emp_id = a.open_emp_id 
inner join (select branch_id from branch where name = 'Woburn Branch') b on b.branch_id = a.open_branch_id 

-- using the same database twice
-- include the branch where account was opened and branch where employee works
select a.account_id, e.emp_id, b.name open_account_branch, b2.name emp_branch from account a 
inner join branch b on b.branch_id = a.open_branch_id 
inner join employee e on e.emp_id = a.open_emp_id 
inner join branch b2 on b2.branch_id = e.assigned_branch_id 

-- self joins 
-- find all the superiors to each employee
select e.fname, e.lname, e_mgr.fname mgr_fname, e_mgr.lname mgr_lname from employee e
inner join employee e_mgr on e_mgr.emp_id = e.superior_emp_id 

select e.emp_id, e.fname, e.lname, e.start_date
from employee e 
inner join product p on e.start_date >= p.date_offered 
and e.start_date <= now()

-- choose all pairs of employees (a,b) = (b,a)
select e.fname, e.lname, 'vs', e2.fname, e2.lname , e.title from employee e
inner join employee e2 on e.emp_id < e2.emp_id
where e.title = 'Teller' and e2.title = 'Teller'

-- return the account id for each nonbusinesss customer (cust_type_cd = 'I')
-- with rederal id and name of product on which account was based, i.e product name
select a.account_id, a.product_cd, c.fed_id, p.name product_name from account a
inner join customer c on a.cust_id = c.cust_id 
inner join product p on p.product_cd = a.product_cd
where c.cust_type_cd = 'I'

-- find all employees whose supervisor is assigned to a different department
-- retrieve employees id, first name and last name
select e.emp_id, e.fname, e.lname, d.name emp_assigned_dpt, mgr.fname mgr_fname, mgr.lname mgr_lname, d2.name mgr_assigned_dept from employee e 
inner join department d on e.dept_id = d.dept_id 
inner join employee mgr on mgr.emp_id = e.superior_emp_id 
inner join department d2 on d2.dept_id = mgr.dept_id 
where mgr.dept_id != e.dept_id 

-- final solution to above
select e.emp_id, e.fname, e.lname from employee e
inner join department d on e.dept_id = d.dept_id 
inner join employee mgr on mgr.emp_id = e.superior_emp_id 
inner join department d2 on d2.dept_id = mgr.dept_id 
where mgr.dept_id != e.dept_id
