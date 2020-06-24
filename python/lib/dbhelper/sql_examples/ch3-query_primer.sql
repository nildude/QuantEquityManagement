select * from department d 

select emp_id, 'ACTIVE' status, emp_id * 3.14159 as tmp, upper(lname) lastnameupper
from employee

select distinct cust_id from account
order by cust_id

-- the from clause
--tables
select e.emp_id, e.fname, e.lname
from (select emp_id, fname, lname, start_date from employee e2 ) e;

-- you can also do
with cte as (select emp_id, fname, lname, start_date from employee e)
select emp_id, fname, lname from cte

--views
create view employee_vw as
select emp_id, fname, lname, date_part('year', start_date) start_year
from employee;

select emp_id, start_year from employee_vw

--TableLinks also called joins
select e.emp_id, e.fname, e.lname, d.name dept_name, e.dept_id
from employee e 
inner join department d on d.dept_id = e.dept_id


--where clause
select emp_id, fname, lname, start_date, title
from employee e 
where title = 'Head Teller'

-- mpre tjam 1 condition
select emp_id, fname, lname, start_date, title
from employee e 
where title = 'Head Teller' and start_date > '2002-05-01'

select emp_id, fname, lname, start_date, title
from employee e 
where (title = 'Head Teller' and start_date > '2002-01-01') or (title = 'Teller' and start_date > '2002-06-01')

-- group by and having clauses
-- get the count of each employee assigned to each department
select * from employee -- notice 1 employee in loan (dept_id = 3), 3 employees in Admin (dept_id 4) 


select d.name, d.dept_id, count(e.emp_id) num_employees
from department d
inner join employee e on e.dept_id = d.dept_id
group by d.name, d.dept_id

-- lets remove loans
select d.name, d.dept_id, count(e.emp_id) num_employees
from department d
inner join employee e on e.dept_id = d.dept_id
group by d.name, d.dept_id
having count(e.emp_id) > 2; -- remove loans as there is only one

-- order by clauses
select * from account a2 

select a.open_emp_id, a.product_cd from account a
order by a.open_emp_id

select a.open_emp_id, a.product_cd, open_date, avail_balance from account a
order by a.open_emp_id, a.product_cd

select a.open_emp_id, a.product_cd, a.open_date, a.avail_balance from account a
order by a.open_emp_id, a.product_cd, a.avail_balance desc, a.open_date

--sorting via expressions
-- sort via last 3 of social
select cust_id, cust_type_cd, city, state, fed_id
from customer
order by right(fed_id, 3)
