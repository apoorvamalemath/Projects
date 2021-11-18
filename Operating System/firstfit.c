#include<stdio.h>
#include<stdlib.h>
void main()
{
int memory[5]={100,500,200,300,600};
int mm[5];
for(int i=0;i<5;i++)
mm[i]=memory[i];
int processnumber[4]={1,2,3,4};
int i,j,processsize[4]={212,412,112,426};
printf("\nmemorylocation   processnumber ");
for(i=0;i<4;i++)
{
for(j=0;j<5;j++)
if(mm[j]>=processsize[i])
{
printf("\n%d               %d ",memory[j],processnumber[i]);
mm[j]=0;
memory[j]=memory[j]-processsize[i];
break;
}
}
int sum=0;
for(i=0;i<5;i++)
sum = sum + memory[i];
printf("\nThe framentation is %d",sum);
}
