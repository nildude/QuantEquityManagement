--chapter 2 practice
create type gender_type as enum('M','F'); -- constraint that checks for 'M' and 'F'
create table person
(person_id smallserial, 
fname varchar(20),
lname varchar(20), 
gender gender_type, 
birth_date date, 
street varchar(30), 
city varchar(20), 
state varchar(20), 
country varchar(20), 
postal_code varchar(20), 
constraint pk_person primary key (person_id)
);

create table favorite_food
(person_id smallserial,
food varchar(20),
constraint pk_favorite_food primary key (person_id, food), -- two primary keys in favorite_food
constraint fk_fav_food_person_id foreign key (person_id)
references person (person_id) -- -- constraints value of person_id to values in person_id in person
); 

select * from favorite_food 
select * from person 

insert into person(
fname, lname, gender, birth_date)
values ('William', 'Turner', 'M', '1972-05-27');

select * from person;  -- notice that person_id 1 is automatically added

insert into favorite_food (person_id, food)
values ((select person_id from person), 'pizza');

select * from favorite_food

insert into favorite_food (person_id, food)
values (1, 'cookies')
insert into favorite_food (person_id, food)
values (2, 'nachos') -- notice the error since id 2 doesn't exist in person table

insert into favorite_food (person_id, food)
values (1, 'nachos')

insert into favorite_food (person_id, food)
values (1, 'nachos') -- we can't do this as two primary keys have same value so wont work

select * from favorite_food
order by food

insert into person 
(fname, lname, gender, birth_date,
street, city, state, country, postal_code)
values ('Susan', 'Smith', 'F', '1975-11-02',
'23 Maple St', 'Arlington', 'VA', 'USA', '20220');

select * from person 

update person 
set street = '1225 Tremont St', city = 'Boston', state = 'MA', country = 'USA', postal_code = 02138
where person_id = '1';



select * from person

-- end of chapter
drop table favorite_food;
drop table person;
